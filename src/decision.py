"""Final match decisions tailored to the challenge metric (macro F0.5 per Source-1 entity).

Two decision policies are available for turning pair probabilities p2 into per-entity match lists:
  threshold  : accept every pair with p2 >= thr (independent per pair)
  expected_f : per Source-1 entity choose the prediction set that maximises the EXPECTED F0.5 of
               that entity, treating its candidate pairs as independent Bernoulli(p2) events.  For
               F-measures under independence the optimum is always a top-k set by probability
               (Jansche 2007; Nan Ye et al. 2012), so only k = 0..n has to be scored.  k = 0 (an
               empty list) is worth 1.0 exactly when no candidate is a true match, which is how
               singletons are handled.
crossfit.py compares both on the held-out split V and stores the winner in model_stage2_cf.json.

decoy_guard: lookalike businesses ("decoys") share name and street with a real Source-1 entity but
carry a house number shifted by a fixed positive offset.  In the training data 79.8% of same-name US
decoys carry one of the offsets below versus 0.13% of true US pairs (0.3% in India); true-match
number noise is typo-like and symmetric.  A matched record whose house number is shifted by one of
these offsets is dropped unless another matched record of the same entity shares that shift.
"""
import numpy as np
import polars as pl

DECOY_OFFSETS = [3, 4, 5, 7, 9, 11, 13, 21]
DECOY_WEAK = [1, 2]
MAX_K = 30


def _pb(p):
    """Poisson-binomial pmf of the number of successes among independent Bernoulli(p)."""
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for x in p:
        d[1:] = d[1:] * (1 - x) + d[:-1] * x
        d[0] *= (1 - x)
    return d


def best_k(p: np.ndarray, beta2: float = 0.25) -> int:
    """p sorted descending -> k maximising E[F_beta] of predicting the top-k."""
    p = np.clip(p[:MAX_K], 0.0, 1.0)
    n = len(p)
    best, best_e = 0, float(np.prod(1 - p))          # k = 0: correct iff nothing is a match
    for k in range(1, n + 1):
        a = _pb(p[:k])                                 # true positives among the predicted
        b = _pb(p[k:])                                 # true matches left out
        ai = np.arange(k + 1)[:, None]; bi = np.arange(n - k + 1)[None, :]
        f = (1 + beta2) * ai / (beta2 * (ai + bi) + k)
        e = float((a[:, None] * b[None, :] * f).sum())
        if e > best_e:
            best, best_e = k, e
    return best


def select_expected_f(pairs: pl.DataFrame, pcol: str = "p2", power: float = 1.0) -> pl.DataFrame:
    """pairs: s1, rid, <pcol>, one row per candidate pair -> the selected (s1, rid) rows."""
    if pairs.height == 0:
        return pairs.select("s1", "rid")
    g = (pairs.select("s1", "rid", (pl.col(pcol).clip(0, 1) ** power).alias("_p"))
         .sort(["s1", "_p"], descending=[False, True])
         .group_by("s1", maintain_order=True).agg("rid", "_p"))
    keep_s1, keep_rid = [], []
    for s1, rids, ps in g.iter_rows():
        k = best_k(np.asarray(ps, dtype=np.float64))
        keep_s1 += [s1] * k; keep_rid += rids[:k]
    return pl.DataFrame({"s1": keep_s1, "rid": keep_rid}, schema={"s1": pairs.schema["s1"], "rid": pairs.schema["rid"]})


def select(pairs: pl.DataFrame, policy: str, thr: float, power: float = 1.0) -> pl.DataFrame:
    if policy == "expected_f":
        return select_expected_f(pairs, "p2", power)
    return pairs.filter(pl.col("p2") >= thr).select("s1", "rid")


def decoy_guard(acc: pl.DataFrame, hq: pl.DataFrame, hs: pl.DataFrame, unlabeled=()) -> pl.DataFrame:
    """acc: accepted (s1, rid); hq: (rid, hnum, country) for records; hs: (s1, hnum) for Source 1.
    Returns acc without the likely decoys (see module docstring).  For countries without training
    labels (`unlabeled`) the weak offsets +1/+2 are dropped as well, since the model is least
    calibrated there."""
    if acc.height == 0:
        return acc
    x = (acc.join(hq.select("rid", pl.col("hnum").cast(pl.Int64, strict=False).alias("hq"), "country"), on="rid", how="left")
            .join(hs.select("s1", pl.col("hnum").cast(pl.Int64, strict=False).alias("hs")), on="s1", how="left")
            .with_columns((pl.col("hq") - pl.col("hs")).alias("d")))
    x = x.with_columns(pl.len().over(["s1", "d"]).alias("n_same_d"))
    strong = pl.col("d").is_in(DECOY_OFFSETS) & (pl.col("n_same_d") == 1)
    weak = pl.col("country").is_in(list(unlabeled)) & pl.col("d").is_in(DECOY_OFFSETS + DECOY_WEAK) & (pl.col("n_same_d") == 1)
    drop = (strong | weak).fill_null(False)
    return x.filter(~drop).select(acc.columns)
