#!/usr/bin/env bash
# End-to-end reproduction: dataset TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# usage: bash run_all.sh <path/to/student_resource/dataset> <work_dir> <output_dir>
set -euo pipefail
DATASET=${1:?dataset dir}; WORK=${2:-work}; OUT=${3:-output}
mkdir -p "$WORK" "$OUT"; WORK=$(realpath "$WORK"); OUT=$(realpath "$OUT"); DATASET=$(realpath "$DATASET")
DATA=$WORK/data; mkdir -p "$DATA"
cd "$(dirname "$0")/src"
python3 convert.py "$DATASET" "$DATA"                                   # TSV -> parquet
python3 learn_translit.py "$DATA" resources/indic_token_dict.json       # Indic->Latin token dictionary (train GT only)
python3 prep.py "$DATA" "$WORK"                                         # normalise names/addresses
python3 run_blocking.py train "$WORK"                                   # candidate generation
python3 run_blocking.py test "$WORK"
GT="$DATA/train_ground_truth.parquet" python3 build_features.py train "$WORK"   # V (3% S1) + 2M random training queries
python3 select_T.py "$WORK"                                             # T (4% S1) for stage 2
GT="$DATA/train_ground_truth.parquet" python3 build_features.py trainT "$WORK"
GT="$DATA/train_ground_truth.parquet" python3 build_features.py trainRest "$WORK"  # all remaining train queries
python3 build_features.py test "$WORK"
python3 crossfit.py "$WORK" all                                         # 2-fold stage-1 ranker + stage-2 matcher (OOF), threshold on V
python3 make_submission.py "$WORK" "$OUT"
