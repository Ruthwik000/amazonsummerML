"""Normalise all source files -> parquet with normalised columns."""
import sys, time
from multiprocessing import Pool
import polars as pl
import normalize as N
from common import NTHREADS


def _proc(args):
    names, addrs, countries = args
    out = {k: [] for k in ["nfull", "ncore", "nalt", "is_domain", "is_dba", "atoks", "anums", "state", "pin", "acity",
                           "pcode", "hnum", "nskel"]}
    for n, a, c in zip(names, addrs, countries):
        full, core, alt, dom, dba = N.normalize_name(n)
        toks, nums, st, pin, cc, pcode, hnum = N.normalize_address(a, c)
        out["nfull"].append(" ".join(full)); out["ncore"].append(" ".join(core)); out["nalt"].append(" ".join(alt))
        out["is_domain"].append(dom); out["is_dba"].append(dba)
        out["atoks"].append(" ".join(toks)); out["anums"].append(" ".join(nums)); out["state"].append(st)
        out["pin"].append(pin); out["acity"].append("|".join(cc))
        out["pcode"].append(pcode); out["hnum"].append(hnum)
        out["nskel"].append(" ".join(N.skeleton(t) for t in core + alt if len(t) >= 3 and not t.isdigit()))
    return out


def prep_file(inp, outp, nproc=NTHREADS, chunk=50000):
    df = pl.read_parquet(inp)
    if df.height == 0:
        empty = {k: pl.Series(k, [], dtype=pl.Int8 if k in ("is_domain", "is_dba") else pl.Utf8) for k in _proc(([], [], [])).keys()}
        pl.concat([df, pl.DataFrame(empty)], how="horizontal").with_columns(pl.lit(0, pl.Int8).alias("is_native")).write_parquet(outp)
        return 0
    names = df["business_name"].fill_null("").to_list(); addrs = df["business_address"].fill_null("").to_list()
    cs = df["country"].fill_null("").to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], cs[i:i + chunk]) for i in range(0, len(names), chunk)]
    del names, addrs, cs
    parts = []
    with Pool(nproc) as p:
        for r in p.imap(_proc, jobs):
            parts.append(pl.DataFrame(r, schema={k: (pl.Int8 if k in ("is_domain", "is_dba") else pl.Utf8) for k in r}))
    del jobs
    out = pl.concat([df, pl.concat(parts)], how="horizontal")
    out = out.with_columns(pl.col("business_name").str.contains(r"[ऀ-ൿ]").cast(pl.Int8).alias("is_native"))
    out.write_parquet(outp)
    return out.height


if __name__ == "__main__":
    data, work = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] if len(sys.argv) > 3 else [f"{s}_source{i}" for s in ("train", "test") for i in (1, 2, 3)]
    import os
    for nm in names:
        if not os.path.exists(f"{data}/{nm}.parquet"):
            print("skip (missing)", nm, flush=True); continue
        t = time.time()
        n = prep_file(f"{data}/{nm}.parquet", f"{work}/{nm}_norm.parquet")
        print(nm, n, f"{time.time()-t:.0f}s", flush=True)
