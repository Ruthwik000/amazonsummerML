# Business Entity Resolution – reproducible pipeline

This pipeline has two stages. Blocking comes first, then a LightGBM pairwise matcher, then a
LightGBM acceptance model that looks at each record's cluster. Everything is trained only on
the provided training data. No external data, APIs, geocoders or pretrained language models
are used.

## Environment

- Python 3.11 (tested on Linux, 2 CPU cores, 7 GB RAM)
- `pip install -r requirements.txt` (on Colab: `requirements-colab.txt`)

All of the models are LightGBM gradient-boosted trees (MIT license), and each is far below
8B parameters.

## Run on Google Colab

Open `colab_run.ipynb` in Colab (*File → Open notebook → GitHub*, or upload it). Fill in the settings
cell (where the code and the dataset zip are, `inference` or `train`) and run all cells. The notebook:
- mounts Drive
- clones the repo
- installs `requirements-colab.txt` (no runtime restart needed)
- runs `run_pipeline.py`, streaming its output
- shows summary stats
- builds `<team>_submission.zip` and downloads it

A CPU runtime is enough. The work directory is on Drive, so after a disconnect you re-run all
cells and the pipeline continues from the step that didn't finish.

## Resumable runner (any machine)

```bash
python3 run_pipeline.py --mode inference --dataset <dataset dir or .zip> --work work --out output [--package --team NAME]
python3 run_pipeline.py --mode train     --dataset <dataset dir or .zip> --work work --out output
```

Finished steps leave markers in `<work>/.done/` and are skipped on re-run. Use `--force` to redo
everything, or `--from <step>` to redo from one step onward. Per-step logs go to `<work>/logs/`.
CPU threads default to all cores; set `N_THREADS` to change that. Output is deterministic: the same
input gives byte-identical files at any thread count.

## Run end-to-end

```bash
bash run_all.sh /path/to/student_resource/dataset work output
# -> output/matching_results.tsv, output/candidate_pairs.tsv
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir /path/to/student_resource/dataset/test
```

**Inference only** (shipped models in `models/`, no training, only the test split is needed):

```bash
bash run_inference.sh /path/to/student_resource/dataset work_infer output
```

**Score** any output against ground truth (e.g. a validation split) with the challenge metric:

```bash
python3 src/evaluate.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --truth <ground_truth.tsv> --source-dir <dir with *_source{1,2,3}.tsv>
```

It prints macro F0.5 (leaderboard metric), macro/micro precision and recall, singleton accuracy,
per-country F0.5, candidates per Source-1 entity, pair completeness (recall ceiling), reduction
ratio, best reachable F0.5 from the candidate set, and format checks.

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
| 8 | `src/make_submission.py` | Scores the test set with the average of the 2 stage-1 models, keeps each record's best Source-1 candidate if p1 ≥ 0.02 (this is `candidate_pairs.tsv`), applies the stage-2 matcher, then the decision policy and decoy guard (`src/decision.py`), and writes both TSVs. |

## Robustness and edge cases

- **Input files** (`convert.py`): UTF-8 BOM, CRLF, ragged rows, whitespace, empty or duplicated
  `entity_id`s, and a missing train split are all handled. Country labels are canonicalised without
  regard to case (`" france"`, `FR`, `USA` …). Unknown labels are kept, because country is an open set.
- **Countries**: nothing is hard-coded to US/India. Per-country files use a filesystem-safe name. A
  country with no Source-2/3 records, or no candidates, still gets one empty row for each of its
  Source-1 entities.
- **Names**: dotted and garbled abbreviations (`S.A.R.L.`, `Eu..R.L.`, `Pvt.Ltd`), glued suffixes,
  and typo'd legal or generic words (`Coropration`, `Limted`, `Inorporated`) are matched by
  optimal-string-alignment distance. Ordinals are no longer "leet"-translated, and French legal
  forms are included.
- **Addresses**: postal codes are extracted for every country (US ZIP, Indian PIN, French code
  postal, and a generic rule for other countries). The state is taken from the right-most matching
  component, so `New York, NY` keeps its city, and it is also found inside `"IL 62704"` or
  `"Austin TX 78701"`. The house/plot number skips postcodes and floor numbers. Typos of street
  types are fixed (`Stret`, `Boulvard`, `Raod`). The full French region list is included, without
  city names.
- **Blocking**: postal-code keys for all countries, plus typo-tolerant consonant-skeleton keys (kinds 5/6).
- **Features** (new, additive, so older models still load): `nskel_tset`, `nskel_tsort`, `pcode_eq`,
  `hnum_eq`, `hnum_diff` (signed house-number offset), `hnum_absdiff`, `num_only_diff`.
- **Decisions** (`decision.py`): the fixed stage-2 threshold, or per-entity expected-F0.5-optimal
  selection. There is also a house-number decoy guard. `crossfit.py` picks the best combination on V
  and stores it in `model_stage2_cf.json`. Override with `POLICY`, `THR2` or `DECOY_GUARD`.

**Candidate generation has two steps:** key retrieval, then the learned candidate ranker (stage 1).
- After retrieval there are 30.5 candidates per Source-1 entity on test.
- After the ranker there are **3.97 per Source-1 entity** (6.9M pairs). This is exactly the set the final matcher (stage 2) scores, and it is what's written to `candidate_pairs.tsv`.

Checks:
- `src/check_candidates.py` does a memory-light check of the candidate file.
- The official validator needs more than 6 GB of RAM when you pass it the candidate file.

Development scripts (`train.py`, `predict_stage1.py`, `train_stage2.py`) hold the earlier T/V experiments. `crossfit.py` reuses `train.read_part`.
