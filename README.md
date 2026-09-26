# Business Entity Resolution – reproducible pipeline

This pipeline has two stages. Blocking comes first, then a LightGBM pairwise matcher, then a
LightGBM acceptance model that looks at each record's cluster. Everything is trained only on
the provided training data. No external data, APIs, geocoders or pretrained language models
are used.

## Environment

- Python 3.11 (tested on Linux, 2 CPU cores, 7 GB RAM)
- `pip install -r requirements.txt`

All of the models are LightGBM gradient-boosted trees (MIT license), and each is far below
8B parameters.

## Run end-to-end

```bash
bash run_all.sh /path/to/student_resource/dataset work output
# -> output/matching_results.tsv, output/candidate_pairs.tsv
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir /path/to/student_resource/dataset/test
```

A full run takes about 3–4 hours on 2 cores. The heaviest steps are feature computation for
about 53M test pairs and stage-1 scoring. Peak memory stays under about 6 GB because every
step works in chunks, per country or per query range.

## Steps (`run_all.sh`)

| # | script | what it does |
|---|--------|--------------|
| 1 | `src/convert.py` | Converts the TSVs to parquet. Everything is read as strings with an explicit tab separator. |
| 2 | `src/learn_translit.py` | Learns a dictionary that maps Indic-script tokens to Latin tokens. It uses the train ground truth, aligning the Latin Source-1 name with the native-script Source-2/3 name by token position. |
| 3 | `src/prep.py` + `src/normalize.py` | Normalises names and addresses. This covers transliteration, accent stripping, leet-speak repair, legal-suffix canonicalisation, DBA/aka/fka/domain parsing, per-country street abbreviations, state/region canonicalisation, number and PIN extraction, and city components. |
| 4 | `src/run_blocking.py` (`keys.py`, `blocking.py`) | Generates candidates. Each Source-2/3 record queries a Source-1 inverted index of hashed composite keys. Candidates are scored by the summed IDF of shared keys, and the top 50 are kept. |
| 5 | `src/build_features.py` (`features.py`) | Computes about 64 pairwise features: rapidfuzz string similarities, token-set, IDF-weighted overlaps, numbers, state, PIN, legal form, blocking scores, and ranks within the query. |
| 6 | `src/select_T.py` + `build_features.py trainT / trainRest` | Computes features for every remaining train query, so each train pair has features. V (3% of Source-1 entities) is the held-out validation set. |
| 7 | `src/crossfit.py` | Stage 1 is the learned candidate ranker: 2 LightGBM models, each trained on one half of the queries (split by hash, V excluded) and scoring the other half. That gives every pair an out-of-fold score. Stage 2 is the matcher, a LightGBM model on each query's best candidate with cluster context and query-to-query similarity. It is trained on 2.5M out-of-fold best pairs from non-V queries, and its threshold is tuned for macro F0.5 on V. |
| 8 | `src/make_submission.py` | Scores the test set with the average of the 2 stage-1 models, keeps each record's best Source-1 candidate if p1 ≥ 0.02 (this is `candidate_pairs.tsv`), applies the stage-2 matcher, and writes both TSVs. |

**Candidate generation has two steps:** key retrieval, then the learned candidate ranker (stage 1).
- After retrieval there are 30.5 candidates per Source-1 entity on test.
- After the ranker there are **3.97 per Source-1 entity** (6.9M pairs). This is exactly the set the final matcher (stage 2) scores, and it is what's written to `candidate_pairs.tsv`.

Checks:
- `src/check_candidates.py` does a memory-light check of the candidate file.
- The official validator needs more than 6 GB of RAM when you pass it the candidate file.

Development scripts (`train.py`, `predict_stage1.py`, `train_stage2.py`) hold the earlier T/V experiments. `crossfit.py` reuses `train.read_part`.
