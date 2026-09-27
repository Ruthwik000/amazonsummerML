#!/usr/bin/env bash
# v17 = v16 + a GPU cross-encoder (distilbert-base-multilingual-cased, Apache-2.0, 135M params, fine-tuned on the
#       provided train pairs only) stacked with the stage-2 score on the uncertain pairs (0.002 < p2 < 0.998) of US and India.
#       France keeps the v13 decisions.
#   Local US+India harness (cross-fitted, before the rescue): v15 0.98787 -> v16 0.98816 -> v17 0.98955.
#   Label-free test checks: v17 drops ~1,900 upward-offset pairs (+k) while -k pairs stay unchanged (= decoys removed).
# usage:
#   bash run_v17.sh <dataset_dir> <work_dir> <output_dir>                      # part 1: v16 pipeline + export for the GPU
#   (run gpu_crossencoder/xenc_colab.ipynb on a GPU, e.g. free Colab T4, with the files in <work_dir>/colab -> xenc_scores.csv)
#   bash run_v17.sh <dataset_dir> <work_dir> <output_dir> <xenc_scores.csv>    # part 2: stacking + post-processing
set -euo pipefail
DATASET=$(realpath "${1:?dataset dir}"); WORK=$(realpath -m "${2:-work}"); OUT=$(realpath -m "${3:-output}"); XS=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd); MD=$HERE/models
cd "$HERE/src"
if [ -z "$XS" ]; then
  bash "$HERE/run_v16.sh" "$DATASET" "$WORK" "$WORK/out_v16_final"
  mkdir -p "$WORK/v16"
  cp "$WORK/cf_preds_v14_T2.parquet" "$WORK/s2v2_v14.json" "$WORK/test_best_s2v2_cf_final.parquet" "$WORK/v16/"
  V15_DIR="$WORK/v16" python3 xenc_export.py "$WORK" "$WORK/colab"        # xtrain / xvt / xtest parquet parts for the GPU
  echo "part 1 done: run gpu_crossencoder/xenc_colab.ipynb with the files in $WORK/colab, then run part 2 with xenc_scores.csv"
  exit 0
fi
XS=$(realpath "$XS")
mkdir -p "$WORK/out_v17" "$WORK/v17_ff" "$WORK/v17_final" "$OUT"
V15_DIR="$WORK/v16" python3 xenc_stack.py "$WORK" eval "$XS"                 # local check (prints the gain)
V15_DIR="$WORK/v16" KSCALE=1.5 python3 xenc_stack.py "$WORK" apply "$XS" "$WORK/out_v17"
python3 decoy_postfilter.py --matching "$WORK/out_v17/matching_results.tsv" --dataset "$DATASET" \
        --out "$WORK/out_v17/matching_results_pf.tsv" --removed "$WORK/out_v17/removed_pf.tsv" --refit --hidden
python3 france_fix2.py "$WORK" "$WORK/out_v17/matching_results_pf.tsv" "$WORK/out_v17/removed_pf.tsv" "$WORK/v17_ff"
python3 rescue_apply.py "$WORK" "$WORK/v17_ff/matching_results.tsv" "$WORK/out_v17/removed_pf.tsv" \
        "$WORK/v17_ff/france_fix2_pairs.parquet" "$WORK/rescue_pred_test.parquet" "$WORK/v17_final" \
        "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_v17/candidate_pairs.tsv"
# final: US + India from v17, France from v13
python3 merge_countries.py "$WORK" "$WORK/out_v13/matching_results.tsv" "$WORK/v17_final/matching_results.tsv" "$OUT/matching_results.tsv" US,India
cp "$WORK/v17_final/candidate_pairs.tsv" "$OUT/candidate_pairs.tsv"      # identical to the v13 candidates
echo "done: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
