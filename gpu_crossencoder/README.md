# GPU cross-encoder (plan for v17)

**Idea.** LightGBM sees hand-made similarity features. A small transformer that reads both texts together, "name | address" of the record and of the Source-1 entity, can catch patterns the features miss: pseudo-names, partial addresses, word swaps. Its score is **stacked** with the LightGBM score p2, and only for uncertain pairs (0.002 < p2 < 0.998). Confident pairs are left unchanged.

- **Model:** `distilbert-base-multilingual-cased` (Apache-2.0, 135M parameters, allowed: ≤ 8B, open licence). It is fine-tuned only on the provided train data.
- **Hardware:** free Google Colab T4 GPU, about 1 hour. The cloud box and the user's PC VM have no GPU.

## Steps
1. **Export pairs** (in the pipeline work dir): `python src/xenc_export.py <work> <out_dir>`
   - `xtrain_part*.parquet`: 646,749 labelled pairs from the training subsets T2–T4 (hard pairs with 0.02 ≤ p1 ≤ 0.999, plus 60K easy ones).
   - `xvt_part*.parquet`: 56,516 uncertain V+T pairs with labels, for the local check and for fitting the stacker.
   - `xtest_part*.parquet`: 1,100,575 uncertain test pairs.
2. **Colab:**
   1. Upload `xenc_colab.ipynb` and set Runtime → T4 GPU.
   2. Upload the 12 parquet files and Run all.
   3. The notebook trains for 2 epochs (batch 128, lr 5e-5, fp16, max length 128) and prints the hold-out AUC.
   4. It downloads `xenc_scores.csv.gz` with columns `rid, s1, score (logit), split ∈ {vt, test}`.
3. **Local check:** `python src/xenc_stack.py <work> eval xenc_scores.csv.gz`
   - It fits a logistic regression on `[logit p2, xenc, product, logit p1]` over the V+T band pairs (2-fold by Source-1, negatives ×2.6 so the scale matches p2).
   - It prints the local leaderboard with and without the cross-encoder, at K×1, ×1.25 and ×1.5.
4. **Apply** (only if the gain is clear): `KSCALE=1 python src/xenc_stack.py <work> apply xenc_scores.csv.gz <out>`
   - It keeps the v15 upward-offset cap and changes only US and India by default (`XENC_COUNTRIES`).
   - Then run the downstream steps (decoy post-filter → `france_fix2.py` → `rescue_apply.py` → `merge_countries.py`, France from v13) and the evaluation checks.

**Expected gain:** unknown until the real scores arrive. A dry run with a fake signal (AUC 0.85, independent of p2) gave +0.0012 locally. A real model is correlated with p2, so expect less.
