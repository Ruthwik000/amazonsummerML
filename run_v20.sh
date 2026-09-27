#!/usr/bin/env bash
# v20 = v19 with a LightGBM stacker (src/xenc_stack_gbm.py) instead of the logistic one, and K x1.75 instead of x1.5.
#   Local US+India harness: v19 0.98969 -> v20 0.98981.
# v19 = v17 with TWO GPU cross-encoders stacked with the stage-2 score on the uncertain pairs of US and India:
#       distilbert-base-multilingual-cased (Apache-2.0, 135M) + xlm-roberta-base (MIT, 278M), both fine-tuned on the
#       provided train pairs only. Also: rescue_apply re-selects the rescued entities with the stacked scores (env BEST).
#   Local US+India harness: v17 0.98955 -> v19 0.98969.   France keeps the v13 decisions.
# usage:
#   bash run_v17.sh <dataset_dir> <work_dir> <output_dir>        # part 1 (v16 pipeline + export for the GPU), same as v17
#   run gpu_crossencoder/xenc_colab.ipynb and gpu_crossencoder/xenc2_xlmr_colab.ipynb on a GPU with the files in <work_dir>/colab
#   bash run_v20.sh <dataset_dir> <work_dir> <output_dir> <xenc_scores.csv> <xenc2_scores.csv>
set -euo pipefail
DATASET=$(realpath "${1:?dataset dir}"); WORK=$(realpath -m "${2:-work}"); OUT=$(realpath -m "${3:-output}")
X1=$(realpath "${4:?xenc_scores.csv}"); X2=$(realpath "${5:?xenc2_scores.csv}")
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/src"
mkdir -p "$WORK/out_v20" "$WORK/v20_ff" "$WORK/v20_final" "$OUT"
V15_DIR="$WORK/v16" python3 xenc_stack_gbm.py "$WORK" eval "$X1,$X2"               # local check (prints the gain)
V15_DIR="$WORK/v16" KSCALE=1.75 python3 xenc_stack_gbm.py "$WORK" apply "$X1,$X2" "$WORK/out_v20"
python3 decoy_postfilter.py --matching "$WORK/out_v20/matching_results.tsv" --dataset "$DATASET" \
        --out "$WORK/out_v20/matching_results_pf.tsv" --removed "$WORK/out_v20/removed_pf.tsv" --refit --hidden
python3 france_fix2.py "$WORK" "$WORK/out_v20/matching_results_pf.tsv" "$WORK/out_v20/removed_pf.tsv" "$WORK/v20_ff"
BEST="$WORK/out_v20/test_best_xenc.parquet" python3 rescue_apply.py "$WORK" "$WORK/v20_ff/matching_results.tsv" "$WORK/out_v20/removed_pf.tsv" \
        "$WORK/v20_ff/france_fix2_pairs.parquet" "$WORK/rescue_pred_test.parquet" "$WORK/v20_final" \
        "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_v20/candidate_pairs.tsv"
# final: US + India from v20, France from v13
python3 merge_countries.py "$WORK" "$WORK/out_v13/matching_results.tsv" "$WORK/v20_final/matching_results.tsv" "$OUT/matching_results.tsv" US,India
cp "$WORK/v20_final/candidate_pairs.tsv" "$OUT/candidate_pairs.tsv"      # identical to the v13 candidates
echo "done: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
