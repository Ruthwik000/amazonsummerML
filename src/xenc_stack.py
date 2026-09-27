"""Stack the GPU cross-encoder score (colab/xenc_colab.ipynb -> xenc_scores.csv.gz) with the stage-2 score p2.

Only pairs in the uncertain band of p2 have a cross-encoder score (xenc_export.py); elsewhere p2 is kept.
In the band the new raw score is a logistic regression on [logit(p2_raw), xenc logit, product, logit(p1)],
fitted on the V+T band pairs (cross-fitted p2, cross-encoder never trained on V/T) with negatives weighted
x2.6 like the stage-2 model, so the result stays on the same scale and the per-country prior shift applies.

usage:
  python xenc_stack.py <work> eval  <xenc_scores.csv.gz>               # local US+India check (2-fold by Source-1)
  python xenc_stack.py <work> apply <xenc_scores.csv.gz> <out_dir>     # test: new p2 -> EV selection -> tsv files
Several cross-encoders: pass the score files comma-separated (a.csv,b.csv); each adds its logit and its product with logit(p2).
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_eval import LocalEval, ev_select, K_V10  # noqa: E402

W, STEP, XS = sys.argv[1], sys.argv[2], sys.argv[3]
XFILES = [f for f in XS.split(",") if f]
XC = [f"xenc{i}" for i in range(len(XFILES))]
V15 = os.environ.get("V15_DIR", f"{W}/v15")
NEGW = 2.6
LO, HI = 0.002, 0.998


def lg(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def feats(p2, x, p1):
    a = lg(p2)
    x = np.asarray(x, dtype=np.float64).reshape(len(a), -1)
    return np.column_stack([a, x, a[:, None] * x, lg(p1)])


def scores(split):
    out = None
    for i, f in enumerate(XFILES):
        d = (pl.read_csv(f, schema_overrides={"rid": pl.UInt32, "s1": pl.UInt32, "score": pl.Float64})
               .filter(pl.col("split") == split).select("rid", "s1", pl.col("score").alias(f"xenc{i}")).unique(["rid", "s1"]))
        out = d if out is None else out.join(d, on=["rid", "s1"])
    return out


def fit(d):
    m = LogisticRegression(C=1.0, max_iter=2000)
    w = np.where(d["label"].to_numpy() == 1, 1.0, NEGW)
    m.fit(feats(d["p2"], d.select(XC).to_numpy(), d["p1"]), d["label"].to_numpy(), sample_weight=w)
    return m


def eval_():
    cf = pl.read_parquet(f"{V15}/cf_preds_v14_T2.parquet")
    x = scores("vt")
    band = cf.join(x, on=["rid", "s1"])
    print("V+T pairs", cf.height, "band pairs with xenc", band.height, "positives", int(band["label"].sum()))
    from sklearn.metrics import roc_auc_score
    print(f"band AUC  p2 {roc_auc_score(band['label'], band['p2']):.4f}   " + "  ".join(f"{c} {roc_auc_score(band['label'], band[c]):.4f}" for c in XC))
    fold = (band["s1"].hash(seed=5) % 2).to_numpy()
    newp = np.empty(band.height)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        m = fit(band.filter(pl.Series(tr)))
        print("fold", k, "coef", np.round(m.coef_[0], 3), "icpt", round(float(m.intercept_[0]), 3))
        bt = band.filter(pl.Series(te))
        newp[te] = m.predict_proba(feats(bt["p2"], bt.select(XC).to_numpy(), bt["p1"]))[:, 1]
    print(f"band AUC  stacked {roc_auc_score(band['label'], newp):.4f}")
    new = band.select("rid", "s1", pl.Series("p2x", newp))
    preds = cf.join(new, on=["rid", "s1"], how="left").with_columns(pl.col("p2x").fill_null(pl.col("p2")))
    le = LocalEval(W)
    for ks in (1.0, 1.25, 1.5):
        kk = {c: K_V10[c] * ks for c in K_V10}
        print(f"--- K x{ks}")
        r0 = le.score(preds, "p2", k=kk)
        r1 = le.score(preds, "p2x", k=kk)
        print(f"    gain {r1['US+India'] - r0['US+India']:+.5f}")


def apply_(out):
    cf = pl.read_parquet(f"{V15}/cf_preds_v14_T2.parquet")
    band = cf.join(scores("vt"), on=["rid", "s1"])
    m = fit(band)
    print("coef", np.round(m.coef_[0], 3), "icpt", round(float(m.intercept_[0]), 3))
    best = pl.read_parquet(f"{V15}/test_best_s2v2_cf_final.parquet")
    x = scores("test")
    s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id", "country"])
    cc = s1.select(pl.col("rid").alias("s1"), "country")
    meta = json.load(open(f"{V15}/s2v2_v14.json"))
    nS1 = cc.group_by("country").len().rename({"len": "n"})
    cnt = best.join(cc, on="s1").group_by("country").len().rename({"len": "cand"}).join(nS1, on="country")
    pos, neg, r0 = meta["train_pos_per_s1"], meta["train_neg_per_s1"], float(meta["ratio"])
    cnt = cnt.with_columns(((pl.col("cand") / pl.col("n") - pos) / neg).alias("ratio_c")).with_columns((r0 / pl.col("ratio_c")).alias("k"))
    ks = float(os.environ.get("KSCALE", "1"))
    if ks != 1.0:
        cnt = cnt.with_columns(pl.when(pl.col("country").is_in(["US", "India"])).then(pl.col("k") * ks).otherwise(pl.col("k")).alias("k"))
    print(cnt.select("country", "ratio_c", "k").to_dicts())
    countries = [c.strip() for c in os.environ.get("XENC_COUNTRIES", "US,India").split(",")]
    b = best.join(x, on=["rid", "s1"], how="left").join(cc, on="s1").join(cnt.select("country", "k"), on="country")
    use = pl.Series([True] * b.height)
    for c in XC:
        use = use & b[c].is_not_null()
    use = use & b["country"].is_in(countries)
    print("test pairs", b.height, "with xenc used", int(use.sum()))
    xv = b.select([pl.col(c).fill_null(0.0) for c in XC]).to_numpy()
    newraw = m.predict_proba(feats(b["p2_raw"], xv, b["p1"]))[:, 1]
    b = b.with_columns(pl.Series("raw2", np.where(use.to_numpy(), newraw, b["p2_raw"].to_numpy())))
    kk = b["k"].to_numpy()
    r2 = b["raw2"].to_numpy()
    adj = r2 * kk / (r2 * kk + 1 - r2)          # every pair (also those without xenc), so KSCALE applies everywhere
    # the v15 cap, as in stage2_v14.predict: strong positive house-number offsets may not exceed the v10 score
    d1 = pl.concat([pl.read_parquet(p, columns=["rid", "s1", "d1"]) for p in sorted(glob.glob(f"{W}/test_best_feats_*.parquet"))])
    v10 = pl.read_parquet(f"{W}/test_best_s2v2_cf_final_v10.parquet").select("rid", "s1", pl.col("p2").alias("p2_v10"))
    b = b.join(d1, on=["rid", "s1"], how="left").join(v10, on=["rid", "s1"], how="left")
    cap = (b["d1"].is_in([3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 21.0]) & b["p2_v10"].is_not_null()).fill_null(False).to_numpy()
    new_p2 = np.where(cap, np.minimum(adj, b["p2_v10"].fill_null(1.0).to_numpy()), adj)
    print("capped pairs", int(cap.sum()))
    b = b.with_columns(pl.Series("p2_base", b["p2"]), pl.Series("p2", new_p2)).drop("country", "k", "raw2", "d1", "p2_v10")
    os.makedirs(out, exist_ok=True)
    b.write_parquet(f"{out}/test_best_xenc.parquet")
    s1id = s1.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
    qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
    acc = ev_select(b, "p2").join(qid, on="rid")
    acc0 = ev_select(b, "p2_base")
    print("accepted pairs base", acc0.height, "-> xenc", acc.height)
    match = (s1id.join(acc.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
                 .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    match.write_csv(f"{out}/matching_results.tsv", separator="\t", quote_style="never")
    cand = b.select("rid", "s1").join(qid, on="rid")
    cp = (s1id.join(cand.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("candidate_entity_ids")), on="s1", how="left")
              .select("source1_entity_id", pl.col("candidate_entity_ids").fill_null("")))
    cp.write_csv(f"{out}/candidate_pairs.tsv", separator="\t", quote_style="never")
    print("written", out)


if __name__ == "__main__":
    {"eval": lambda: eval_(), "apply": lambda: apply_(sys.argv[4])}[STEP]()
