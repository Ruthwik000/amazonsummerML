#!/usr/bin/env bash
# v16 = v15 + one more training subset T4 (10% of the train Source-1 entities, 2.94M stage-2 rows in total)
#       + a K x1.5 prior shift for US and India (KSCALE=1.5; local harness optimum).
#   Local US+India harness (cross-fitted matcher, before the rescue): v15 0.98787 -> v16 (K x1) 0.98802 -> v16 (K x1.5) 0.98816.
#   Label-free test checks: no rise in accepted upward decoy offsets; France keeps the v13 decisions as in v15.
# usage: bash run_v16.sh <path/to/student_resource/dataset> <work_dir> <output_dir>
set -euo pipefail
DATASET=$(realpath "${1:?dataset dir}"); WORK=$(realpath -m "${2:-work}"); OUT=$(realpath -m "${3:-output}")
HERE=$(cd "$(dirname "$0")" && pwd); MD=$HERE/models
mkdir -p "$WORK/data" "$OUT" "$WORK/out_v10" "$WORK/out_v12" "$WORK/out_v13" "$WORK/out_v14" "$WORK/out_v14_ff" "$WORK/out_v16us"
cd "$HERE/src"
# ---- data, blocking, features
python3 convert.py "$DATASET" "$WORK/data"
python3 prep.py "$WORK/data" "$WORK"
python3 run_blocking.py train "$WORK"
python3 run_blocking.py test "$WORK"
N_TRAIN_Q=1 GT="$WORK/data/train_ground_truth.parquet" python3 build_features.py train "$WORK"   # V (3% of S1)
python3 select_T.py "$WORK"                                                                    # T (4% of S1)
GT="$WORK/data/train_ground_truth.parquet" python3 build_features.py trainT "$WORK"
python3 build_features.py test "$WORK"
# ---- v13 pipeline (v10 matcher, decoy post-filter, France correction, rescue) -> France rows
python3 stage2_v2.py "$WORK" "$MD" prep_train
NEG_RATIO=2.6 python3 stage2_v2.py "$WORK" "$MD" train_cf
S2KEY=cf_final SELECT=ev PRIOR_ADJ=1 python3 stage2_v2.py "$WORK" "$MD" predict "$WORK/out_v10"
cp "$WORK/test_best_s2v2_cf_final.parquet" "$WORK/test_best_s2v2_cf_final_v10.parquet"
python3 decoy_postfilter.py --matching "$WORK/out_v10/matching_results.tsv" --dataset "$DATASET" \
        --out "$WORK/out_v10/matching_results_v10.tsv" --removed "$WORK/out_v10/removed_v10.tsv" --refit --hidden
python3 france_fix2.py "$WORK" "$WORK/out_v10/matching_results_v10.tsv" "$WORK/out_v10/removed_v10.tsv" "$WORK/out_v12"
python3 rescue_queries.py "$WORK"
python3 rescue_cands.py "$WORK" train "$WORK/rescue_q_train.parquet" "$WORK/rescue_cand_train.parquet"
python3 rescue_cands.py "$WORK" test "$WORK/rescue_q_test.parquet" "$WORK/rescue_cand_test.parquet"
python3 rescue_score.py "$WORK" train
python3 rescue_score.py "$WORK" predict
python3 rescue_apply.py "$WORK" "$WORK/out_v12/matching_results.tsv" "$WORK/out_v10/removed_v10.tsv" \
        "$WORK/out_v12/france_fix2_pairs.parquet" "$WORK/rescue_pred_test.parquet" "$WORK/out_v13" \
        "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_v10/candidate_pairs.tsv"
# ---- v16 matcher: extra training subsets T2, T3, T4
python3 select_T2.py "$WORK" T2 7 17
SUB=T2 python3 build_features.py trainT2 "$WORK"
python3 prep_T2.py "$WORK" "$MD" T2
python3 select_T2.py "$WORK" T3 17 27
SUB=T3 python3 build_features.py trainT2 "$WORK"
python3 prep_T2.py "$WORK" "$MD" T3
python3 select_T2.py "$WORK" T4 27 37
SUB=T4 python3 build_features.py trainT2 "$WORK"
python3 prep_T2.py "$WORK" "$MD" T4
python3 stage2_v14.py "$WORK" "$MD" train                     # prints the local cross-fit check
for C in US India France; do python3 extra_test.py "$WORK" "$C"; done
KSCALE=1.5 python3 stage2_v14.py "$WORK" "$MD" predict "$WORK/out_v14"   # one process per country, prior shift, cap, EV selection
python3 decoy_postfilter.py --matching "$WORK/out_v14/matching_results.tsv" --dataset "$DATASET" \
        --out "$WORK/out_v14/matching_results_pf.tsv" --removed "$WORK/out_v14/removed_pf.tsv" --refit --hidden
python3 france_fix2.py "$WORK" "$WORK/out_v14/matching_results_pf.tsv" "$WORK/out_v14/removed_pf.tsv" "$WORK/out_v14_ff"
python3 rescue_apply.py "$WORK" "$WORK/out_v14_ff/matching_results.tsv" "$WORK/out_v14/removed_pf.tsv" \
        "$WORK/out_v14_ff/france_fix2_pairs.parquet" "$WORK/rescue_pred_test.parquet" "$WORK/out_v16us" \
        "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_v14/candidate_pairs.tsv"
# ---- final: US + India from v16, France from v13
python3 merge_countries.py "$WORK" "$WORK/out_v13/matching_results.tsv" "$WORK/out_v16us/matching_results.tsv" "$OUT/matching_results.tsv" US,India
cp "$WORK/out_v16us/candidate_pairs.tsv" "$OUT/candidate_pairs.tsv"      # identical to the v13 candidates
echo "done: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
