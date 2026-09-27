#!/usr/bin/env bash
# v21 = v20 + the rescue step's candidates re-scored with a GPU cross-encoder (DistilBERT, gpu_crossencoder/xenc3_rescue_colab.ipynb):
#       logistic regression on [logit p_rescue, cross-encoder logit, product] (src/xenc_rescue.py), US + India only.
#   Local US+India harness (with the rescue): v20 0.99046 -> v21 0.99067; rescue top-1 correct 16,194 -> 17,134 of 19,305.
# usage:
#   bash run_v17.sh <dataset_dir> <work_dir> <output_dir>                    # part 1 (v16 pipeline + GPU export)
#   run gpu_crossencoder/xenc_colab.ipynb and xenc2_xlmr_colab.ipynb (band pairs) on a GPU
#   bash run_v20.sh <dataset_dir> <work_dir> <output_dir> <xenc_scores.csv> <xenc2_scores.csv>
#   python3 src/xenc_export_rescue.py <work_dir> <work_dir>/colab3 ; run gpu_crossencoder/xenc3_rescue_colab.ipynb with those files + xtrain parts
#   bash run_v21.sh <dataset_dir> <work_dir> <output_dir> <xenc3_extra_scores.csv>
set -euo pipefail
DATASET=$(realpath "${1:?dataset dir}"); WORK=$(realpath -m "${2:-work}"); OUT=$(realpath -m "${3:-output}"); X3=$(realpath "${4:?xenc3_extra_scores.csv}")
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/src"
mkdir -p "$WORK/v21_final" "$OUT"
LOCAL_PREDS="$WORK/v20_local_gbm.parquet" python3 xenc_rescue.py "$WORK" eval "$X3" || true     # local check (needs the v20 local preds)
python3 xenc_rescue.py "$WORK" apply "$X3" "$WORK/rescue_pred_test_x.parquet"
BEST="$WORK/out_v20/test_best_xenc.parquet" python3 rescue_apply.py "$WORK" "$WORK/v20_ff/matching_results.tsv" "$WORK/out_v20/removed_pf.tsv" \
        "$WORK/v20_ff/france_fix2_pairs.parquet" "$WORK/rescue_pred_test_x.parquet" "$WORK/v21_final" \
        "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_v20/candidate_pairs.tsv"
python3 merge_countries.py "$WORK" "$WORK/out_v13/matching_results.tsv" "$WORK/v21_final/matching_results.tsv" "$OUT/matching_results.tsv" US,India
cp "$WORK/v21_final/candidate_pairs.tsv" "$OUT/candidate_pairs.tsv"
echo "done: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
