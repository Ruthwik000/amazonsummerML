"""Score the test set with both stages and write the two submission files.

usage: python make_submission.py <work_dir> <out_dir>
Models are read from <work_dir> when present (fresh training run), otherwise from $MODEL_DIR or
the repository's models/ folder (shipped models, inference-only run).
Environment overrides: THR2 (stage-2 threshold), POLICY (threshold|expected_f), DECOY_GUARD (0|1).
"""
import sys, glob, json, os, time
import numpy as np
import polars as pl
import lightgbm as lgb
import stage2
import decision
from common import safe, NTHREADS
from train import read_part

W, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
REPO_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MD = os.environ.get("MODEL_DIR") or next((d for d in (W, REPO_MODELS) if os.path.exists(f"{d}/model_stage2_cf.txt")
                                          or os.path.exists(f"{d}/model_stage2.txt")), W)
CF = os.path.exists(f"{MD}/model_stage2_cf.txt")   # cross-fitted models (2 stage-1 fold models, averaged)
if CF:
    m1_meta = json.load(open(f"{MD}/model_stage1_cf.json"))
    m1_files = [f"{MD}/model_stage1_f0.txt", f"{MD}/model_stage1_f1.txt"]
    m2_meta = json.load(open(f"{MD}/model_stage2_cf.json")); m2_file = f"{MD}/model_stage2_cf.txt"
else:
    m1_meta = json.load(open(f"{MD}/model_stage1.json")); m1_files = [f"{MD}/model_stage1.txt"]
    m2_meta = json.load(open(f"{MD}/model_stage2.json")); m2_file = f"{MD}/model_stage2.txt"
THR = float(os.environ.get("THR2", m2_meta["thr"]))
POLICY = os.environ.get("POLICY", m2_meta.get("policy", "threshold"))
POWER = float(m2_meta.get("power", 1.0))
GUARD = os.environ.get("DECOY_GUARD", str(int(m2_meta.get("decoy_guard", True)))) == "1"
TRAIN_COUNTRIES = m2_meta.get("train_countries")
print("models", os.path.abspath(MD), "policy", POLICY, "thr", THR, "decoy_guard", GUARD, flush=True)
countries = pl.scan_parquet(f"{W}/test_s1.parquet").select(pl.col("country").unique()).collect()["country"].to_list()
parts_of = lambda c: sorted(glob.glob(f"{W}/feat/test_{safe(c)}_part*.parquet"))
P1_MIN = 0.02  # stage 2 is only trained on best pairs with p1 >= P1_MIN; below it the pair is rejected

# ---- stage 1 scores for every candidate pair
pred_path = f"{W}/pred_test{'_cf' if CF else ''}.parquet"
if not os.path.exists(pred_path):
    m1s = [lgb.Booster(model_file=f) for f in m1_files]
    res = [pl.DataFrame(schema={"rid": pl.UInt32, "s1": pl.UInt32, "p1": pl.Float32})]
    for c in countries:
        for p in parts_of(c):
            d = read_part(p)
            if d.height == 0:
                continue
            X = d.select(m1_meta["cols"]).to_numpy().astype(np.float32)
            p1 = np.mean([m.predict(X, num_threads=NTHREADS) for m in m1s], axis=0)
            res.append(d.select(pl.col("rid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.Series("p1", p1).cast(pl.Float32)))
        print("stage1", c, f"{time.time()-t0:.0f}s", flush=True)
    pl.concat(res).write_parquet(pred_path)
print("stage1 ready", f"{time.time()-t0:.0f}s", flush=True)

# ---- stage 2 on each query's best candidate (per country: clusters never cross countries)
best_path = f"{W}/test_best_pred{'_cf' if CF else ''}.parquet"
m2 = lgb.Booster(model_file=m2_file)
s1dups = stage2.s1_duplicates(pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "country", "ncore", "atoks"]))
outs = [pl.DataFrame(schema={"rid": pl.UInt32, "s1": pl.UInt32, "p1": pl.Float32, "p2": pl.Float32})]
for c in countries:
    parts = parts_of(c)
    if not parts:
        print("stage2", c, "no candidate pairs -> all Source-1 entities of this country stay empty", flush=True)
        continue
    qtext = pl.scan_parquet(f"{W}/test_q.parquet").filter(pl.col("country") == c).select("rid", "entity_id", "ncore", "atoks", "anums").collect()
    pred_c = pl.scan_parquet(pred_path).join(qtext.select("rid").lazy(), on="rid").collect()
    if pred_c.height == 0:
        continue
    cf = pl.concat([read_part(p).select("rid", "s1", "a_tset", "nf_tset") for p in parts])
    best = stage2.build(pred_c, qtext, cf, s1dups)
    del cf, pred_c, qtext
    for p in parts:
        f = best.join(read_part(p), on=["rid", "s1"], how="inner")
        if f.height == 0:
            continue
        outs.append(f.select(pl.col("rid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("p1").cast(pl.Float32),
                             pl.Series("p2", m2.predict(f.select(m2_meta["cols"]).to_numpy().astype(np.float32), num_threads=NTHREADS)).cast(pl.Float32)))
    del best
    print("stage2", c, f"{time.time()-t0:.0f}s", flush=True)
best = pl.concat(outs).unique(["rid"], keep="first")   # one best Source-1 candidate per record
best = best.with_columns(pl.when(pl.col("p1") >= P1_MIN).then(pl.col("p2")).otherwise(0.0).alias("p2"))
best.write_parquet(best_path)
print("stage2 done", f"{time.time()-t0:.0f}s", flush=True)

# ---- decisions
cand_pairs = best.filter(pl.col("p1") >= P1_MIN)
acc = decision.select(cand_pairs, POLICY, THR, POWER)
n_before = acc.height
if GUARD:
    q = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "hnum", "country"])
    s = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "hnum"]).rename({"rid": "s1"})
    unl = [c for c in countries if TRAIN_COUNTRIES is not None and c not in TRAIN_COUNTRIES]
    acc = decision.decoy_guard(acc, q, s, unl)
    print(f"decoy guard removed {n_before - acc.height} pairs (countries without training labels: {unl})", flush=True)

# ---- write files (aggregated on entity ids so every Source-1 entity gets exactly one row)
s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id"]).rename({"rid": "s1", "entity_id": "source1_entity_id"})
qid = (pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
       .filter(pl.col("entity_id").str.contains(r"^S[23]-")))          # only Source-2/3 ids are allowed in the lists
s1_rows = s1.select("source1_entity_id").unique(maintain_order=True)


def write(pairs: pl.DataFrame, col: str, path: str):
    lists = (pairs.select("rid", "s1").join(qid, on="rid").join(s1, on="s1")
             .group_by("source1_entity_id").agg(pl.col("entity_id").unique().sort().str.join(",").alias(col)))
    out = s1_rows.join(lists, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


match = write(acc, "matched_entity_ids", f"{OUT}/matching_results.tsv")
# candidate_pairs.tsv = exactly the pairs the final matcher (stage 2) scores: each record's
# top-ranked Source-1 candidate from the learned candidate ranker (stage 1), kept if p1 >= P1_MIN
cp = write(cand_pairs, "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")

# ---- self-check (the same rules the official validator enforces)
assert match.height == s1_rows.height == cp.height, "row count"
assert match["source1_entity_id"].n_unique() == match.height, "duplicate Source-1 rows"
m_ex = acc.join(cand_pairs.select("rid", "s1"), on=["rid", "s1"], how="anti").height
assert m_ex == 0, "matched pairs outside the candidate set"
print("policy", POLICY, "threshold", THR, "matched pairs", acc.height, "S1 with >=1 match",
      match.filter(pl.col("matched_entity_ids") != "").height, "of", match.height,
      "| candidates", cand_pairs.height, f"({cand_pairs.height / max(match.height, 1):.2f} per S1)", f"{time.time()-t0:.0f}s")
