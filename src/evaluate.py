"""Score a submission against ground truth with the challenge metric, plus blocking statistics.

usage: python evaluate.py --matching output/matching_results.tsv --truth <ground_truth.tsv>
                          [--candidate output/candidate_pairs.tsv] [--source-dir <dir with *_source{1,2,3}.tsv>]

Reports
  * macro F0.5 over ALL Source-1 entities (the leaderboard metric; singletons included)
  * macro precision / recall, micro precision / recall / F0.5
  * singleton accuracy (empty predicted for an entity with no true match) and false-merge rate
  * F0.5 on entities that do have matches
  * per-country breakdown (when --source-dir is given)
  * blocking: candidates per Source-1 entity, pair completeness (recall ceiling), reduction ratio,
    and the best macro F0.5 reachable from the candidate set
  * format checks that mirror the official validator rules
To score a validation split, pass a truth file that only contains the validation Source-1 rows.
"""
import argparse
import os
import sys

import polars as pl


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, truncate_ragged_lines=True).with_columns(pl.all().fill_null(""))


def id_lists(df, col):
    k = df.columns[0]
    return {r[0].strip(): [x.strip() for x in r[1].split(",") if x.strip()] for r in df.select(k, col).iter_rows()}


def f05(tp, n_pred, n_true):
    """Per-entity (F0.5, precision, recall) exactly as the scorer defines F0.5.  Precision is None
    for an empty prediction and recall is None for an empty truth (excluded from those averages)."""
    p = tp / n_pred if n_pred else None
    r = tp / n_true if n_true else None
    if n_pred == 0 and n_true == 0:
        return 1.0, p, r
    if tp == 0:
        return 0.0, p, r
    return 1.25 * p * r / (0.25 * p + r), p, r


def format_issues(pred, s1_ids, valid_other, name):
    issues = []
    rows = pred.columns
    if len(rows) != 2:
        issues.append(f"{name}: expected 2 columns, got {rows}")
    keys = pred[rows[0]].to_list()
    if len(keys) != len(set(keys)):
        issues.append(f"{name}: duplicate source1_entity_id rows")
    missing = set(s1_ids) - set(keys)
    if missing:
        issues.append(f"{name}: {len(missing)} Source-1 entities missing")
    extra = set(keys) - set(s1_ids)
    if extra:
        issues.append(f"{name}: {len(extra)} unknown Source-1 ids")
    n_dup = n_bad = n_unknown = 0
    for k, ids in id_lists(pred, rows[1]).items():
        n_dup += len(ids) - len(set(ids))
        n_bad += sum(not (x.startswith("S2-") or x.startswith("S3-")) for x in ids)
        if valid_other is not None:
            n_unknown += sum(x not in valid_other for x in ids)
    if n_dup:
        issues.append(f"{name}: {n_dup} duplicate ids inside lists")
    if n_bad:
        issues.append(f"{name}: {n_bad} ids without S2-/S3- prefix")
    if n_unknown:
        issues.append(f"{name}: {n_unknown} ids not present in Source 2/3")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--source-dir", default=None, help="directory holding <split>_source{1,2,3}.tsv")
    a = ap.parse_args()

    truth = id_lists(read_tsv(a.truth), "matched_entity_ids")
    mdf = read_tsv(a.matching)
    pred = id_lists(mdf, mdf.columns[1])
    s1_ids = list(truth)
    country, valid_other, n_other_by_c = {}, None, {}
    if a.source_dir:
        files = sorted(os.listdir(a.source_dir))
        f1 = [f for f in files if f.endswith("source1.tsv")]
        if f1:
            d1 = read_tsv(os.path.join(a.source_dir, f1[0]))
            country = dict(zip(d1["entity_id"].str.strip_chars().to_list(), d1["country"].str.strip_chars().to_list()))
        others = [read_tsv(os.path.join(a.source_dir, f)) for f in files if f.endswith("source2.tsv") or f.endswith("source3.tsv")]
        if others:
            o = pl.concat([x.select("entity_id", "country") for x in others])
            valid_other = set(o["entity_id"].str.strip_chars().to_list())
            n_other_by_c = dict(o.group_by("country").len().iter_rows())

    # ---------------- matching quality
    stats = {"all": [], "singleton": [], "matched": []}
    per_c = {}
    tp_all = np_all = nt_all = 0
    for s in s1_ids:
        t, p = set(truth[s]), set(pred.get(s, []))
        tp = len(t & p)
        f, pr, rc = f05(tp, len(p), len(t))
        tp_all += tp; np_all += len(p); nt_all += len(t)
        stats["all"].append((f, pr, rc))
        stats["singleton" if not t else "matched"].append((f, len(p)))
        per_c.setdefault(country.get(s, "?"), []).append(f)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    micro_p = tp_all / np_all if np_all else 0.0
    micro_r = tp_all / nt_all if nt_all else 0.0
    micro_f = 1.25 * micro_p * micro_r / (0.25 * micro_p + micro_r) if micro_p + micro_r else 0.0
    sing = stats["singleton"]
    print("=" * 64)
    print(f"Source-1 entities evaluated        : {len(s1_ids)}")
    print(f"  with >=1 true match / singletons : {len(stats['matched'])} / {len(sing)}")
    print(f"MACRO F0.5 (leaderboard metric)    : {mean([x[0] for x in stats['all']]):.5f}")
    print(f"  macro precision (non-empty pred) : {mean([x[1] for x in stats['all'] if x[1] is not None]):.5f}")
    print(f"  macro recall (non-empty truth)   : {mean([x[2] for x in stats['all'] if x[2] is not None]):.5f}")
    print(f"  F0.5 on entities with matches    : {mean([x[0] for x in stats['matched']]):.5f}")
    print(f"  singleton accuracy               : {mean([x[0] for x in sing]):.5f}  "
          f"(false merges on singletons: {sum(1 for _, n in sing if n)})")
    print(f"MICRO precision / recall / F0.5    : {micro_p:.5f} / {micro_r:.5f} / {micro_f:.5f}")
    print(f"  predicted pairs / true pairs / TP: {np_all} / {nt_all} / {tp_all}")
    if len(per_c) > 1 or "?" not in per_c:
        for c, fs in sorted(per_c.items()):
            print(f"  country {c:<10} n={len(fs):<7} macro F0.5={mean(fs):.5f}")

    # ---------------- blocking quality
    if a.candidate:
        cdf = read_tsv(a.candidate)
        cand = id_lists(cdf, cdf.columns[1])
        n_c = sum(len(v) for v in cand.values())
        covered = sum(len(set(truth[s]) & set(cand.get(s, []))) for s in s1_ids)
        # best reachable macro F0.5 = predict exactly the true matches that survived blocking
        ub = mean([f05(len(set(truth[s]) & set(cand.get(s, []))), len(set(truth[s]) & set(cand.get(s, []))), len(truth[s]))[0]
                   for s in s1_ids])
        not_sub = sum(len(set(pred.get(s, [])) - set(cand.get(s, []))) for s in pred)
        print("-" * 64)
        print(f"candidate pairs                    : {n_c}")
        print(f"candidates per Source-1 entity     : {n_c / max(len(cand), 1):.3f}")
        print(f"pair completeness (recall ceiling) : {covered / max(nt_all, 1):.5f}")
        if valid_other is not None:
            full = sum(n_other_by_c.get(country.get(s, "?"), 0) for s in s1_ids)  # same-country cross product
            full_any = len(s1_ids) * len(valid_other)
            print(f"reduction ratio vs same-country   : {1 - n_c / max(full, 1):.6f}")
            print(f"reduction ratio vs all S1 x S2/S3  : {1 - n_c / max(full_any, 1):.6f}")
        print(f"best macro F0.5 reachable          : {ub:.5f}")
        print(f"matched ids not in candidates      : {not_sub}")

    # ---------------- format
    issues = format_issues(mdf, s1_ids, valid_other, "matching_results")
    if a.candidate:
        issues += format_issues(cdf, s1_ids, valid_other, "candidate_pairs")
    print("-" * 64)
    print("FORMAT: PASS" if not issues else "FORMAT ISSUES:\n  " + "\n  ".join(issues))
    print("=" * 64)
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
