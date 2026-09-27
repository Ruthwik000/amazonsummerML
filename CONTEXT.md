# CONTEXT: read this first (for a new chat / AI assistant)

Last updated: 2026-09-27, about 16:30 IST.

## 1. What we are doing
- **Competition:** Amazon ML Challenge 2026, problem "Business Entity Resolution", on Unstop.
- **Team:** Ekdant.exe
  - Gurram Ruthwik Goud
  - Nagasri Pavan Kumar Dharmapuri
  - Manne Lakshmi Varaprasad
  - Pattaswamy Vishwak Yashasree
- **Deadline:** 2026-09-28 00:00 IST (= 2026-09-27 18:30 UTC).
- **Final ranking:** uses the **private** leaderboard, so **the last upload must be a good file**.
- **Task:** Source-1 is a reference table of businesses. Source-2 and Source-3 are noisy copies of them, plus **decoys**: near-copies that belong to nobody. For every Source-1 entity, list the Source-2/3 records that are the same business.
  - Countries: US and India in train; US, India and **France** in test. France has no train labels.
  - Every Source-2/3 record belongs to at most one Source-1 entity.
- **Metric:** macro F0.5 per Source-1 entity, where precision counts more than recall.
  - A singleton (an entity with no true match) scores 1 only when its prediction is empty.
  - Every false positive hurts a lot. The best set is chosen per entity (expected-F0.5 selection).
- **Files to upload:**

  | file | content |
  |---|---|
  | `matching_results.tsv` | columns `source1_entity_id`, `matched_entity_ids` (comma list, may be empty) |
  | `candidate_pairs.tsv` | columns `source1_entity_id`, `candidate_entity_ids` |

  Final package layout: `output/` (the two TSVs), `code/business_entity_resolution/` (src, README.md, requirements.txt), `Documentation_template.md`.
- **Rules:**
  - No external data or APIs.
  - Pretrained models are allowed only if MIT/Apache licensed and ≤ 8B parameters.
  - The official validator is `student_resource/utils/validate_submission.py`. Its flag is `--candidate` (not `--candidates`).

## 2. Where we are now
| version | leaderboard (public) | note |
|---|---|---|
| v12 | 0.985429 | France fix |
| **v15** | **0.986916** (rank 174) | **current best, safe file** |
| top teams | about 0.9915 | about 173 teams are above us |

- **The safe file is `amazon-ml/submission_v15/`**, the same as `amazon-ml/output/`.
  - matching_results MD5 `26bc8c9dc4cc9224dd4115ca260d063f`, candidate_pairs MD5 `5260904cfcf1d6c583b9b934d8fef0ce`.
  - The validator passes.
- The implied per-country leaderboard is US/India ≈ 0.988 and France ≈ 0.981.
- The local US+India harness (`evaluation/evaluate.py`) predicted the v15 gain correctly (+0.0015).

## 3. What is running or ready right now
1. **v16 matcher:** the v15 matcher plus one more training subset T4 (2.94M rows).
   - Local cross-fit: 0.98802, vs v15's 0.98787 (+0.00015, small).
   - It is being predicted and post-processed in the cloud.
   - Upload it only if its combined gain is clearly positive.
2. **K×1.5 prior shift for US/India** (`KSCALE=1.5`): +0.00015 locally. It is not applied yet, because it may not transfer to the test.
3. **GPU cross-encoder (main hope for a bigger jump):**
   - A multilingual DistilBERT (Apache-2.0, 135M params) fine-tuned on (record, Source-1) text pairs.
   - It runs on the free Google Colab T4. The notebook and data are in `amazon-ml/colab_gpu/`; see `gpu_crossencoder/README.md`.
   - Its score is stacked with the LightGBM score p2, only for uncertain pairs (0.002 < p2 < 0.998), by `src/xenc_stack.py`.
   - A test with a fake signal of AUC 0.85 on those pairs gave about +0.0012 locally. The real gain is unknown until the scores arrive.

## 4. Plan and decision rules
- **Test locally before any upload.** Uploads are limited, so the user does not want wasted ones.
  - US/India: `evaluation/evaluate.py`.
  - France, which has no labels: `evaluation/prof_diff.py` and `evaluation/sym_check.py`.
  - Upload only if the local gain is clearly > 0 (≥ about +0.0003) and the label-free test checks show no extra decoys.
- If a new upload scores **below 0.986916**, re-upload `submission_v15` before the deadline.
- **Order:**
  1. Cross-encoder scores from Colab.
  2. `python src/xenc_stack.py <work> eval <scores>`, which gives the local gain.
  3. If positive: `apply`, then the downstream steps (decoy post-filter → france_fix2 → rescue_apply → merge France from v13), then validate and deliver as `submission_v17`.
- France stays on the v13 decisions unless a France-specific check shows a gain.

## 5. How the user wants to work (important)
- Explanations must be **short and simple**. The user is not a native English speaker.
- **Never make zip files.** Save results directly into the connected folder (`C:\Users\ruthw\OneDrive\Desktop\amazon-ml`).
- Test everything offline first: *"i cannot upload always"*.
- Don't spend the user's credits on browsing when the user has already pasted the text.
- Never type passwords or credentials. The user uploads to Unstop themselves.

## 6. Where things live
- **Device folder `amazon-ml/`:**
  - `submission_vN/`: the files uploaded for each version.
  - `output/`: the current final files.
  - `code/business_entity_resolution/`: this code.
  - `colab_gpu/`: notebook + data for the cross-encoder.
  - `student_resource/`: the dataset and the official validator.
  - `_archive/`: old files moved out of the code folder.
- **Cloud workspace** (temporary, lost when the session ends): `/home/claude/p2` (code + `work/` with all intermediate parquet files, many GB).
  - A new chat has to rebuild `work/` with `run_v15.sh`, which takes about 5 hours on 2 cores.
- **Claude project "amazon-ml":** notes `claude/amazon_ml_er_progress.md` and `claude/amazon_ml_strategy.md`.
- **Also read:**
  - `APPROACHES_AND_LESSONS.md`: everything we tried and what we learnt, including mistakes. Read it before proposing ideas, because many obvious ideas were already tried.
  - `README.md`: how the pipeline runs.
