# CONTEXT: read this first (for a new chat / AI assistant)

Last updated: 2026-09-27, 20:35 IST (see the changelog at the bottom).

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
- **Organizer update (2026-09-27): `candidate_pairs.tsv` and the code that makes it are reviewed for the final ranking.**
  - Blocking must scale (no all-pairs comparison).
  - A **smaller candidate set per Source-1 entity ranks higher**, beyond the leaderboard.
  - Ours is 3.98 per entity (52.9M after retrieval → 6.89M), with 97.7% of true pairs kept locally; see README.
  - Do not grow the candidate set. Every matched id must be inside it.
  - We decided NOT to shrink it further with the stage-2 score, even though p2 < 0.001 would cut it to 3.74. That would be cosmetic: stage 2 already runs only on these candidates, and it would make the candidates look like "the matches".
- **Rules:**
  - No external data or APIs.
  - Pretrained models are allowed only if MIT/Apache licensed and ≤ 8B parameters.
  - The official validator is `student_resource/utils/validate_submission.py`. Its flag is `--candidate` (not `--candidates`).

## 2. Where we are now
| version | leaderboard (public) | note |
|---|---|---|
| v12 | 0.985429 | France fix |
| v15 | 0.986916 (rank 174) | |
| v17 | 0.988917 | |
| v19 | 0.989000 | |
| v20 | 0.989191 | best confirmed LB |
| **v21** | final submission | **`amazon-ml/output/` = `submission_v21/`** (local +0.0002 over v20) |
| top teams | about 0.9915 | about 173 teams are above us |

- **The safe file is `amazon-ml/submission_v20/`**, the same as `amazon-ml/output/`, copied at 20:41 IST.
  - matching_results MD5 `48020b9ca8399c67e08c80807b7bed51`, candidate_pairs MD5 `5260904cfcf1d6c583b9b934d8fef0ce`.
  - The validator passes.
  - Documentation_template.md describes v20.
- The implied per-country leaderboard for v15 was US/India ≈ 0.988 and France ≈ 0.981. For v17, US/India ≈ 0.990 if France is unchanged.
- The local US+India harness (`evaluation/evaluate.py`) predicted the v15 gain correctly (+0.0015).

## 3. What is running or ready right now
1. **v17 was uploaded: LB 0.988917** (+0.0020 over v15). It is now in `output/`.
   - What it is: v16 + a GPU cross-encoder.
     - The model is `distilbert-base-multilingual-cased`, fine-tuned on Colab T4 in about 35 minutes.
     - It is stacked with p2 on the uncertain pairs of US and India.
     - France is from v13.
   - Local US+India: v15 0.98787 → v16 0.98816 → **v17 0.98955** (+0.0017 vs v15). US alone is 0.9902.
   - Band AUC: p2 0.9789, cross-encoder 0.9462, stacked 0.9878.
   - Label-free test checks:
     - Accepted +k offset pairs drop by about 1,900, while −k pairs stay the same. The removed pairs are decoys.
     - The pairs it adds have a true-match profile.
   - The official validator PASSes.
   - MD5: matching `3a2d58d32a36bdf3456745593dd10e99`, candidates `5260904cfcf1d6c583b9b934d8fef0ce` (unchanged, 3.98 per S1).
   - Code: `run_v17.sh` (part 1: v16 + export; Colab notebook; part 2: stacking + post-processing). The stacker coefficients are in `models/xenc_stack_v17.json`.
2. **v16** in `amazon-ml/submission_v16/`: T4 rows + K×1.5, local 0.98816, validator PASS (MD5 `b69929590b9ecd9c7c722e1be0c38a53`). It is superseded by v17.
3. **v18 (cross-encoder for France too) was REJECTED.** The label-free checks showed it removes about 9.8K true-profile France pairs and adds 5.7K pairs with about 32% decoys. France stays on v13.
4. **v19 READY in `amazon-ml/submission_v19/`:** DistilBERT + XLM-R cross-encoders stacked together, plus the rescue BEST fix.
   - Local 0.98969, vs 0.98955 for v17 (+0.00014, small). XLM-R alone scored 0.98952.
   - Label-free checks are neutral to positive: no extra upward-offset pairs, and the added pairs have a true profile.
   - The validator PASSes. MD5: matching `ad5b8ac70ba932b158d536e58ad75fe2`, candidates `5260904cfcf1d6c583b9b934d8fef0ce`.
   - Advised as the next upload. If it scores below 0.988917, re-upload v17.
   - Code: `run_v19.sh` and `models/xenc_stack_v19.json`.
   - The rescue extra files (`xextra`) were not scored by the user, so the rescue re-scoring (`xenc_rescue.py`) is not used.
5. **Bug fixed (for v19):** `rescue_apply.py` used the stage-2 scores without the cross-encoder for the ~20K entities it re-selects. It now reads the scores from env `BEST` (the `test_best_xenc.parquet` of the xenc run). Only 268 rows change, so the effect is tiny.
6. **Deep loss analysis (v17 local, oracle gains):**
   - Empty-address twin records: about 0.006 in total (buckets b/c/e [EA]). This is not fixable.
   - No candidates [addr] 0.0016 (the rescue recovers about 0.0007).
   - True S1 not in candidates [addr] 0.0007, not first [addr] 0.0005, rejected [addr] 0.0007.
   - All false positives 0.0010.
   - To reach 0.99, the LB needs +0.0011, which is about +0.0013 on US/India. That is hard; a realistic target tonight is 0.9893–0.9897.

## 4. Plan and decision rules
- **Test locally before any upload.** Uploads are limited, so the user does not want wasted ones.
  - US/India: `evaluation/evaluate.py`.
  - France, which has no labels: `evaluation/prof_diff.py` and `evaluation/sym_check.py`.
  - Upload only if the local gain is clearly > 0 (≥ about +0.0003) and the label-free test checks show no extra decoys.
- If a new upload scores **below the best LB so far**, re-upload the best earlier file (**v20 = 0.989191**) before the deadline.
- **Order (as of 17:35 IST):**
  1. ✅ Cross-encoder done → v17.
  2. ✅ v17 uploaded: LB 0.988917, and copied into `output/`.
  3. ~~France with the cross-encoder (v18)~~: rejected by the label-free checks.
  4. Optional: second cross-encoder (XLM-R) on Colab, then stack both, then v19 if the local gain is clear.
  5. Before the deadline, make sure `amazon-ml/output/` + the code folder + `Documentation_template.md` match the best uploaded file.
- France stays on the v13 decisions unless a France-specific check shows a gain.

## 5. How the user wants to work (important)
- **Keep `CONTEXT.md`, `README.md` and `APPROACHES_AND_LESSONS.md` up to date continuously**, after every result, upload or decision. Also copy CONTEXT and LESSONS to the Claude project (`claude/`), and add a line to the changelog below.
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
  - `DEEP_ANALYSIS.md`: where the remaining errors are and what could reach 0.99+. **Read this before trying new ideas.**

## Changelog (newest last, IST)
- 09-27 15:30: v15 uploaded, LB 0.986916 (rank 174).
- 09-27 15:40: Colab cross-encoder files are in `amazon-ml/colab_gpu/`. Waiting for the user to run them.
- 09-27 16:05: Code folder cleaned: old files moved to `amazon-ml/_archive/code_old_2026-09-27/`. Added `evaluation/`, `gpu_crossencoder/`, CONTEXT.md and APPROACHES_AND_LESSONS.md.
- 09-27 16:20: v16 built: local 0.98816, validator PASS, in `amazon-ml/submission_v16/`. Recommended as the next upload.
- 09-27 16:25: Organizer note: candidate set size counts. The candidate funnel was added to Documentation_template.md and README (3.98 per S1).
- 09-27 16:35: Added `run_v16.sh`, the v16 model, and the `KSCALE` option in `stage2_v14.py`.
- 09-27 17:20: Colab done by the user (`amazon-ml/xenc_scores.csv`). The cross-encoder stacking gives local +0.0014 over v16.
- 09-27 17:30: v17 built, checked and validated → `amazon-ml/submission_v17/`. Recommended upload. Added `run_v17.sh` and `models/xenc_stack_v17.json`.
- 09-27 17:45: v18 (cross-encoder for France) rejected by the label-free checks. A LightGBM stacker was +0.00008 over the logistic one (noise), so it is not used.
- 09-27 18:00: Optional XLM-R notebook placed in `amazon-ml/colab_gpu2/`. `xenc_stack.py` now accepts several score files.
- 09-27 18:07: **v17 LB = 0.988917** (+0.0020 over v15). Copied into `amazon-ml/output/`. Documentation_template.md and README updated.
- 09-27 18:35: Deep loss analysis done. XLM-R notebook + rescue extra files prepared, and the user asked to run them. `rescue_apply.py` BEST fix. New `src/xenc_rescue.py`.
- 09-27 19:45: XLM-R scores arrived. v19 (both models) local 0.98969 → `submission_v19/`, validator PASS. Recommended upload (small gain).
- 09-27 20:05: **v19 LB = 0.989** (+0.00008 over v17). Copied into `output/`, and Documentation_template.md now describes v19.
- 09-27 20:10: Analysis for 0.99 (see APPROACHES L16):
  - No ID or row leak (ID mod-m checks at chance level).
  - India has about 2,750 more accepted +1/+2 offset pairs on the test than the local reference predicts, but they are spread thin: no cell has a decoy share above 0.3.
  - France rejects 40% of −1/−2 pairs (US 3%), but that is only 338 pairs.
  - A web research agent found no public top-solution tricks.
  - Conclusion: 0.99 is not reachable before the deadline without new signal.
- 09-27 20:40: Deep analysis written to `DEEP_ANALYSIS.md`. New scripts in `evaluation/deep/` (loss_breakdown, miss_kinds, twin_rates, offset_cells, leak_checks), all tested. Key facts: 60% of the loss is empty-address coin flips, France has 1.7× more of them, and 0.99 would need days of GPU work (re-ranking, dense retrieval, bigger cross-encoder, France pseudo-labels).
- 09-27 20:30: **v20 READY** in `amazon-ml/submission_v20/`: v19 with a LightGBM stacker (both cross-encoders + pair decoy/noise features, `src/xenc_stack_gbm.py`) and K×1.75.
  - Local 0.98981 vs 0.98969 for v19.
  - Label-free checks: the added pairs have a true profile (decoy share 0–0.08), and there is no rise in strong-offset pairs.
  - Validator PASS. MD5: matching `48020b9ca8399c67e08c80807b7bed51`, candidates unchanged.
  - Advised upload. If it scores below 0.989000, re-upload v19. `output/` stays on v19 until v20's LB is known.
  - A second internet search (GitHub, blogs, LinkedIn) found nothing above 0.969 in public. One repo claims the LB top was 0.9906 on 09-26.
- 09-27 20:50: Checked even deeper after v20: stacker variants, per-country K, rescue odds, empty-address segment odds, and the kinds of empty-address misses. All noise (±0.00005), so nothing to add. Details in DEEP_ANALYSIS.md. **v20 is the last improvement.**
- 09-27 20:41: **v20 LB = 0.989191** (+0.00019 over v19; local predicted +0.00012, so the LB again gained more than local). Copied into `output/`; Documentation_template.md describes v20.
- 09-27 20:50: Asked the user for one more short Colab run: `amazon-ml/colab_gpu3/xenc3_rescue_colab.ipynb` + 6 data files → `xenc3_extra_scores.csv.gz`.
  - It re-scores the rescue step's top-5 candidates with DistilBERT (`src/xenc_rescue.py`).
  - A fake-signal dry run gave about +0.0001. The cloud script `work/build_v21.sh` then builds v21 = v20 + the new rescue (tested with fake scores: validator PASS).
- 09-27 20:55: Also checked the empty-prediction rate (US 5.67% vs 5.51% true singletons; the wrong-empties mostly have no true pair in the candidates) and the remaining false positives (mostly empty-address near-twins, digit-drop decoys, and pseudo-name decoys at exact addresses). None of these is fixable quickly.
- 09-27 21:10: Checked France with the cross-encoder stack split by empty address (worse), removing the US cap (worse by the symmetry reference: ~530 decoys added), and rescue recalibration (no gain). **v20 stays best.** The only pending small idea is the Colab rescue run (xenc3).
- 09-27 22:20: Pseudo-name sibling propagation tested (no gain: unmatched alias copies are decoys). A third web search found no generator code and no public solution above ~0.98. **v20 (0.989191) is final.**
- 09-27 22:15: **v21 built** from the user's Colab rescue scores (`xenc3_extra_scores.csv.gz`, uploaded in chat).
  - Rescue AUC 0.9788 → 0.9933; top-1 correct 16,194 → 17,134.
  - Local with rescue 0.99046 → 0.99067 (+0.0002).
  - Official validator PASS. MD5: matching `538ad002a7dd8479dce480f663bc6263`, candidates `f1029a9b5d4b1c747719779f99e68007` (6,892,248 pairs, 3.98 per S1).
  - The link to the user's PC dropped. Patches `patch_m/c_v21_from_v20.tsv.gz` + `apply_patch2.py` are ready (tested: they rebuild v21 exactly from v20) and go into `amazon-ml/submission_v21/` when it reconnects.
  - Documentation for v21 is prepared, to switch after its LB score.
- 09-27 23:15: **v21 chosen as the FINAL submission** (user). `output/` = submission_v21 (MD5 matching `538ad002…`, candidates `f1029a9b…`). Documentation_template.md describes v21. The code folder is ready for GitHub: README run steps for the full v21 chain, `gpu_crossencoder/` notebooks + requirements-gpu.txt, and `.gitignore`.
