"""Convert the challenge TSVs to parquet.  usage: python convert.py <dataset_dir> <data_out_dir>

Everything is read as strings with an explicit tab separator and no quoting.  Defensive cleaning:
  * UTF-8 BOM / stray whitespace in the header, CRLF line endings, ragged rows
  * surrounding whitespace in every field; literal "null"/"NaN" ids are treated as empty
  * country labels canonicalised case-insensitively (" france", "FR", "USA" ...) while unknown
    labels are kept as they are (country is an open set)
  * rows without an entity_id are dropped, duplicated entity_ids keep their first row
  * a missing split (e.g. no train files for an inference-only run) is skipped
"""
import sys, os
import polars as pl

COUNTRY_ALIASES = {
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US", "united states": "US", "united states of america": "US",
    "america": "US",
    "india": "India", "in": "India", "ind": "India", "bharat": "India", "republic of india": "India",
    "france": "France", "fr": "France", "fra": "France", "republique francaise": "France", "république française": "France",
}
NULLS = ["null", "none", "nan", "<null>", "n/a"]


def canon_country(col: pl.Expr) -> pl.Expr:
    low = col.str.strip_chars().str.to_lowercase()
    return low.replace_strict(COUNTRY_ALIASES, default=col.str.strip_chars(), return_dtype=pl.Utf8)


def read_tsv(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, truncate_ragged_lines=True,
                     encoding="utf8-lossy")
    df = df.rename({c: c.replace("﻿", "").strip() for c in df.columns})
    return df.with_columns(pl.all().fill_null("").str.replace_all(r"[\r\n]+", " ").str.strip_chars())


def clean_source(df: pl.DataFrame, name: str) -> pl.DataFrame:
    for c in ("entity_id", "business_name", "business_address", "country"):
        if c not in df.columns:
            print(f"  WARNING {name}: column {c!r} missing, filled with empty strings", flush=True)
            df = df.with_columns(pl.lit("").alias(c))
    n0 = df.height
    df = df.filter((pl.col("entity_id") != "") & ~pl.col("entity_id").str.to_lowercase().is_in(NULLS))
    df = df.unique("entity_id", keep="first", maintain_order=True)
    if df.height != n0:
        print(f"  {name}: dropped {n0 - df.height} rows with empty / duplicated entity_id", flush=True)
    return df.with_columns(canon_country(pl.col("country")).alias("country"))


def clean_gt(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns
    df = df.rename({cols[0]: "source1_entity_id", cols[1]: "matched_entity_ids"}) if len(cols) >= 2 else \
        df.rename({cols[0]: "source1_entity_id"}).with_columns(pl.lit("").alias("matched_entity_ids"))
    ids = pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().str.strip_chars()).list.eval(
        pl.element().filter(pl.element() != "")).list.unique(maintain_order=True).list.join(",")
    return (df.select("source1_entity_id", ids.alias("matched_entity_ids"))
              .group_by("source1_entity_id", maintain_order=True)
              .agg(pl.col("matched_entity_ids").str.join(","))
              .with_columns(pl.col("matched_entity_ids").str.strip_chars(",")))


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for split in ("train", "test"):
        for name in [f"{split}_source1", f"{split}_source2", f"{split}_source3"] + (["train_ground_truth"] if split == "train" else []):
            path = f"{src}/{split}/{name}.tsv"
            if not os.path.exists(path):
                print("skip (missing)", path, flush=True)
                continue
            df = read_tsv(path)
            df = clean_gt(df) if name.endswith("ground_truth") else clean_source(df, name)
            df.write_parquet(f"{out}/{name}.parquet")
            extra = "" if name.endswith("ground_truth") else f" countries={dict(df.group_by('country').len().iter_rows())}"
            print(name, df.shape, extra, flush=True)
