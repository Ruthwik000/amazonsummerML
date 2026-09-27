# Business Entity Resolution: Amazon ML Challenge 2026 (team Ekdant.exe)

This project matches noisy business records (Source-2 and Source-3) to a reference table (Source-1) in the US, India and France. The test set also contains deliberate **decoys**: near-copies of a business that belong to nobody.

- **Final submission: v15.** Public leaderboard 0.986916. The metric is macro F0.5 per Source-1 entity.
- **New here?** Read [`CONTEXT.md`](CONTEXT.md) (status, plan, rules) and [`APPROACHES_AND_LESSONS.md`](APPROACHES_AND_LESSONS.md) (everything tried, what worked, what failed).

## How it works (one paragraph)
1. **Blocking.** Composite keys over name and address tokens, scoped by country, find about 50 candidate Source-1 entities per record.
2. **Stage-1 ranker.** A LightGBM ranker keeps each record's single best candidate (p1 ≥ 0.02).
3. **Stage-2 matcher.** A decoy-aware LightGBM matcher scores that pair. It uses name/address similarity, house-number offset features, legal-form swaps, decoy-word scores and cluster context. It is trained with decoys weighted ×2.6, because the test has about 3× more decoys than train.
4. **Selection.** Scores are shifted to each country's test prior. For every Source-1 entity, the set of records with the highest **expected F0.5** is chosen, and "nothing" is one of the options.
5. **Post-processing.** A decoy post-filter, label-free France rules and a rescue search for records without any candidate. France keeps the v13 decisions.

## Folder layout
```
README.md                  this file
CONTEXT.md                 current status, plan, decision rules, user preferences (read first)
APPROACHES_AND_LESSONS.md  version history, findings, lessons, mistakes, tried-and-dropped ideas
requirements.txt
run_all.sh                 first-time training of the stage-1 ranker (crossfit.py) + transliteration dictionary (+ the v1 baseline)
run_v15.sh                 FINAL pipeline: dataset -> output/matching_results.tsv + candidate_pairs.tsv
models/                    trained models used for the submitted files
src/                       pipeline code (table below)
evaluation/                evaluation engine: local mini-leaderboard + label-free test checks + light validator
gpu_crossencoder/          optional GPU text model (Colab notebook) and how to stack it (v17 plan)
```

## Run
```bash
pip install -r requirements.txt
# models/ already holds the stage-1 ranker; to rebuild it from scratch run run_all.sh first
# (it writes work/model_stage1_* and work/model_stage2_cf.*; copy them into models/)
bash run_v15.sh /path/to/student_resource/dataset work output
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir /path/to/student_resource/dataset/test
```
- A full run takes about 4–5 hours on 2 CPU cores, with peak RAM under about 6 GB (everything is chunked).
- The official validator needs more than 6 GB for the candidate file. `evaluation/check_submission.py` does the same checks with little memory.

## Pipeline steps (`run_v15.sh`)
| # | script | what it does |
|---|---|---|
| 1 | `src/convert.py` | TSV → parquet (all strings, tab separator) |
| 2 | `src/prep.py`, `src/normalize.py` | normalise names/addresses: Indic transliteration (`src/resources/indic_token_dict.json`, learned by `learn_translit.py`), accents, leet-speak, legal forms, street abbreviations, states, numbers, PIN |
| 3 | `src/run_blocking.py` (`keys.py`, `blocking.py`) | composite-key blocking, per-kind frequency caps, top 50 by summed IDF |
| 4 | `src/build_features.py` (`features.py`), `src/select_T.py` | about 64 pairwise features for the held-out subsets V (3% of train Source-1), T (4%) and the test |
| 5 | `src/stage2_v2.py prep_train / train_cf / predict` (`stage2.py`, `decoy_features.py`, `train.py`, `metrics.py`) | v10 decoy-aware matcher: out-of-fold stage-1 scores, cluster context, negatives ×2.6, prior shift, expected-F0.5 selection |
| 6 | `src/decoy_postfilter.py --refit --hidden` | removes pairs in an entity's positive house-number offset cells |
| 7 | `src/france_fix2.py` (`france_fix.py`) | France: reject business-type-word swaps, accept noise suffixes (count-profile rules) |
| 8 | `src/rescue_queries.py`, `rescue_cands.py`, `rescue_score.py`, `rescue_apply.py` | TF-IDF rescue search for records with no candidate; re-runs the selection for those entities |
| 9 | `src/select_T2.py`, `build_features.py trainT2`, `prep_T2.py` | extra training subsets T2, T3 (10% of train Source-1 each) |
| 10 | `src/stage2_v14.py train / predict` (`extra_feats.py`, `extra_test.py`) | v15 matcher: + positional house-number features, 3.9× rows; upward-offset pairs capped at the v10 score; one process per country |
| 11 | steps 6–8 again on the v15 scores, then `src/merge_countries.py` | final file: US+India from v15, France from v13 |

`candidate_pairs.tsv` is exactly the set the final decision layer scores: each record's best entity (6.87M pairs) plus 22.7K rescued pairs, 3.98 per Source-1 entity. Every matched id is in it.

## Evaluation engine (`evaluation/`, details in `evaluation/README.md`)
| tool | what it tells you | labels needed |
|---|---|---|
| `evaluate.py` (engine: `src/local_eval.py`) | local leaderboard for US+India at the test decoy density; agrees with the LB within about 0.0004 | train labels |
| `prof_diff.py` | per-country decoy share of the pairs a new file adds or removes (count profile) | none (works for France) |
| `sym_check.py` | accepted +k vs −k house-number offsets on the test: excess = decoys accepted | none |
| `check_submission.py` | format checks with low memory (row count, ids, every match inside the candidates) | none |

## Models and licences
- The submitted v15 uses only LightGBM models (MIT), trained on the provided train data. It uses no external data, APIs or pretrained models.
- The optional cross-encoder step (`gpu_crossencoder/`) fine-tunes `distilbert-base-multilingual-cased` (Apache-2.0, 135M parameters, under the 8B limit), only on the provided train data.
- `models/`:
  - `model_stage1_f0/f1.txt` + `model_stage1_cf.json`: stage-1 ranker (2 folds).
  - `model_stage2_cf.*`: the base feature list read by `stage2_v2.py`.
  - `s2v2_cf_final.txt`: v10 matcher. `s2v2_cf_final_dw.txt`: decoy-word model.
  - `s2v2_cf_v15.txt` + json: v15 matcher.
  - `model_rescue.txt`: rescue model.
  - The run rebuilds all stage-2 and rescue models in `work/`; these copies document the submitted file.
