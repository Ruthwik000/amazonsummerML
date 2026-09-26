#!/usr/bin/env bash
# Inference only (no training): test TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# using the models in ./models (or $MODEL_DIR).  Only the test split is needed.
# usage: bash run_inference.sh <path/to/student_resource/dataset> <work_dir> <output_dir>
set -euo pipefail
DATASET=${1:?dataset dir}; WORK=${2:-work_infer}; OUT=${3:-output}
mkdir -p "$WORK" "$OUT"; WORK=$(realpath "$WORK"); OUT=$(realpath "$OUT"); DATASET=$(realpath "$DATASET")
DATA=$WORK/data; mkdir -p "$DATA"
cd "$(dirname "$0")/src"
python3 convert.py "$DATASET" "$DATA"                                   # TSV -> parquet (missing train split is skipped)
python3 prep.py "$DATA" "$WORK" test_source1 test_source2 test_source3  # normalise names/addresses
python3 run_blocking.py test "$WORK"                                    # candidate generation
python3 build_features.py test "$WORK"                                  # pairwise features
python3 make_submission.py "$WORK" "$OUT"                               # stage 1 + stage 2 + decisions
python3 check_candidates.py "$OUT" "$DATASET/test"                      # memory-light format check
