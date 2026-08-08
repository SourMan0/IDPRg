# Pipeline outputs manifest (ESM → Rg → sequence-logo interpretation)

Tracks every artifact written by each stage of the leakage-free re-run. Rows marked **OLD** were produced by the leaky pre-PCA pipeline and are preserved for diffs (`*_leaky` suffix where renamed). Rows marked **NEW** are produced by the leak-free Pipeline-based code.

## Stage 0 — Raw inputs (unchanged)

| File | Producer | Description |
|------|----------|-------------|
| `training/all_points.csv` | upstream (cleanData.py) | 167 sequences + 20 Rg targets (raw, normalised, regressed) |
| `training/inliers.csv` | upstream (cleanData.py) | 163 inliers (excludes rows 114, 125, 137, 163) |
| `esmScripts/esm_embeddings/esm6layer.pt` | upstream | (N, 7, 320) ESM-6 hidden states |
| `esmScripts/esm_embeddings/esm12layer.pt` | upstream | (N, 13, 480) ESM-12 hidden states |

## Stage 1 — Pre-baked PCA features (DEPRECATED under new pipeline)

These were the leakage source: `StandardScaler().fit_transform` and `PCA().fit_transform` were applied to all 167 points *before* the regressors ever saw a train/test split.

| File | Producer (OLD) | Status |
|------|----------------|--------|
| `esmScripts/6esmPCA/layer{0..6}_pca{10,20,50,100,167}.npy` | `6esmPCAMake.py` | **DEPRECATED** — no longer consumed |
| `esmScripts/12esmPCA/layer{0..12}_pca{10,20,50,100,167}.npy` | `12esmPCAMake.py` | **DEPRECATED** — no longer consumed |
| `esmScripts/6esmPCA/*.csv`, `esmScripts/12esmPCA/*.csv` | (same) | **DEPRECATED** |

## Stage 2 — Sweep losses (model selection)

| File | Producer | Description | Status |
|------|----------|-------------|--------|
| `losses/esmLosses{1..5}_leaky.csv` | `esmRegressionsAllSeeds.py` (OLD) | Test R²/RMSE per (label-split × ESM mode × layer × PCA × model × split) for 5 seeds, with leakage | archived |
| `esmScripts/esmLosses{1..5}.csv` | `esmRegressionsAllSeeds.py` (NEW, multiprocess) | Same matrix produced with Pipeline(StandardScaler+PCA+model) so the test fold is held out before any preprocessing fit. Same 10-column schema; `Principal Components` column now records the actually-fit PCA dim (capped to 0.8·n_train so PCA fits inside each CV fold). | **IN PROGRESS** (background job) |
| `esmScripts/sweep.log` | `esmRegressionsAllSeeds.py` (NEW) | stdout/stderr of the background sweep, with per-200-task progress + ETAs | **NEW** |
| `esmScripts/best_configs.csv` | `selectBestConfigs.py` (NEW) | mean-R² ranking across seeds × splits for the leak-free sweep, filtered to a chosen target / regr-out / points subset. Helper to pick the new best (layer, PCA, model) for the interpretation refit. | **NEW** |

## Stage 3 — Interpretation models (chosen-arch refits)

`prepareForInterpretation.py` is now a CLI accepting `--lasso-esm/--lasso-layer/--lasso-pca` and `--krr-*`. Defaults reproduce the previously chosen configs (ESM-12 L6 PCA50 Lasso, ESM-6 L3 PCA100 KRR); override once the new sweep results pick different winners.

| File | Producer | Description | Status |
|------|----------|-------------|--------|
| `interpretingESM/esm_pca.joblib` | `prepareForInterpretation.py` (OLD) | Standalone PCA(50) fit on all 163 inlier ESM-12 layer 6 embeddings (leaky for CV α selection) | **DEPRECATED** |
| `interpretingESM/lasso.joblib` | `prepareForInterpretation.py` (OLD) | `GridSearchCV(Lasso)` over leaky PCA features | **DEPRECATED** |
| `interpretingESM/esm_pca2.joblib` | `prepareForInterpretation.py` (OLD) | Standalone PCA(100) on ESM-6 layer 3 inliers | **DEPRECATED** |
| `interpretingESM/krr.joblib` | `prepareForInterpretation.py` (OLD) | `GridSearchCV(KernelRidge)` over leaky PCA features | **DEPRECATED** |
| `interpretingESM/lasso_pipeline.joblib` | `prepareForInterpretation.py` (NEW) | Single `Pipeline([StandardScaler, PCA, Lasso])` fit on all 163 inliers, α selected via leak-free CV over the pipeline (PCA refit per fold). `predict(X)` applies scaler→PCA→Lasso in one shot. | **PENDING REFIT** |
| `interpretingESM/krr_pipeline.joblib` | `prepareForInterpretation.py` (NEW) | Single `Pipeline([StandardScaler, PCA, KernelRidge])`, (α,γ) selected via leak-free CV | **PENDING REFIT** |
| Final chosen configs (Lasso) | `selectBestConfigs.py` | **ESM-6 layer 1, PCA=100, Lasso** — clean mean R²=0.322 (n=15 across seeds×splits); inner-CV R² at refit time 0.354; α=0.00464 | ✓ |
| Final chosen configs (KRR)   | `selectBestConfigs.py` | **ESM-6 layer 1, PCA=100, Kernel Ridge** — clean mean R²=0.352; inner-CV R² at refit time 0.378; α=0.1, γ=1e-4 | ✓ |
| Old configs (for comparison) | (now stored only as `*_leaky.joblib`) | ESM-12 L6 PCA50 Lasso clean R²=0.187; ESM-6 L3 PCA100 KRR clean R²=0.280; ESM-12 L4 PCA190 GPR (the "best R²=0.554" claim) clean R²=0.098 | replaced |

## Stage 4 — Sliding-window occlusion outputs

| File | Producer | Description | Status |
|------|----------|-------------|--------|
| `interpretingESM/allEffects4.pkl`, `allFragments4.pkl` | `tryingDifferentType2.py` (OLD) | ΔRg + fragment lists for window sizes 1..10 using leaky `krr.joblib`+`esm_pca2.joblib`. Used by `findMotifs3.py` | **TO BE REGENERATED** |
| `interpretingESM/allEffects4_leaky.pkl`, `allFragments4_leaky.pkl` | (rename of OLD before rerun) | preserved for diff | will rename |
| `interpretingESM/allEffects4.pkl`, `allFragments4.pkl` | `tryingDifferentType2.py` (NEW) | Same, but using leak-free `*_pipeline.joblib` | **NEW** |
| `interpretingESM/allEffects3.pkl`, `allFragments3.pkl` | older sliding-window variant | preserved unchanged (different model path) |
| `interpretingESM/allEffects.pkl`, `allFragments.pkl`, `allMaskedPositions.pkl` | `applySlidingWindowToAll.py` | requires `esm_gpr.joblib` which isn't produced by `prepareForInterpretation.py`; unchanged |
| `interpretingESM/allEffects2.pkl`, `allFragments2.pkl`, `allMaskedPositions2.pkl`, `baselines.pkl` | `applyingOtherSlidingWindowToAll.py` | same — uses `esm_gpr.joblib` |

## Stage 4b — FG-Nup case study (sliding occlusion on 4 Nups)

`exploringFGNups.py` slides a k=3 occlusion window along each of Nup98, Nup49, Nup153 NUS domain, and Nup153 NUL domain, asks the leak-free KRR pipeline how the predicted Rg changes, and plots per-residue ΔRg with F (Phe) and G (Gly) highlighted.

| File | Status |
|------|--------|
| `interpretingESM/exploringFGNups.py` | refactored to load `krr_pipeline.joblib` directly, use ESM-6 layer 1, drop duplicate `plt.bar` bug, save a single 4-panel figure |
| `interpretingESM/fgnup_per_residue_drg.pdf` / `.png` | **NEW** 4-panel figure (one row per Nup) |

The script also prints the reference Rg and ΔRg range per Nup at runtime.

## Stage 5 — Final journal logos

| File | Producer | Description | Status |
|------|----------|-------------|--------|
| `interpretingESM/baseline_logos_fragment_sizes.pdf` | `findMotifs3.py` (OLD inputs) | 3×2 grid: expansion vs compaction logos for fragment sizes 3, 6, 10 | **TO BE REGENERATED** |
| `interpretingESM/baseline_logos_fragment_sizes_leaky.pdf` | rename of OLD | preserved for diff |
| `interpretingESM/baseline_logos_fragment_sizes.pdf` | `findMotifs3.py` (NEW inputs) | Same, regenerated from leak-free pipelines and pickles |
| `interpretingESM/baseline_sequence_logos.pdf` | earlier variant of `findMotifs3.py` | unchanged unless rerun |
| `interpretingESM/baseline_mean_dRg_by_residue.pdf`, `mean_dRg_by_residue.pdf`, `reference_mean_dRg_by_residue.pdf` | exploratory mean-effect plots | unchanged unless rerun |

## Stage P — Physical-features pipeline (parallel to ESM stages above)

The canonical chain for the physical-features journal figure was:
`extractFeaturesFromPoints3St.py → protein_features3St2.csv → doRegressions3.py → pfeatureLosses*St3.csv` and `interpretFeatures2.py → reference_ridge_coefficients_sorted.pdf`.

**Leak**: [extractFeaturesFromPoints3St.py:121-125](IDPRg/physicalFeatures/extractFeaturesFromPoints3St.py#L121-L125) (OLD) ran `StandardScaler().fit_transform` across all 167 sequences before saving the CSV. Both the sweep and the interpretation script therefore consumed pre-scaled features whose μ, σ pooled the test points. (No PCA in this pipeline, so the leak amounts to ≤2% drift in per-feature std and barely moves R².)

**Fix**: feature CSV stores raw values; both sweep and interpretation wrap `Pipeline([StandardScaler, model])` so the scaler is refit per train fold during CV.

| File | Status |
|------|--------|
| `physicalFeatures/protein_features3St2_leaky.csv` | OLD, pre-scaled, archived |
| `physicalFeatures/protein_features3St2.csv` | NEW, raw |
| `physicalFeatures/pfeatureLosses{1..5}St3_leaky.csv` (in both `physicalFeatures/` and `losses/`) | archived |
| `physicalFeatures/pfeatureLosses{1..5}St3.csv` | NEW leak-free sweep, schema unchanged (8 cols) |
| `physicalFeatures/pfeature_sweep.log` | NEW |
| `pfeaturesInterpretation/baseline_ridge_coefficients_leaky.pdf`, `reference_ridge_coefficients_sorted_leaky.pdf` | archived |
| `pfeaturesInterpretation/reference_ridge_coefficients_sorted.pdf` | **NEW journal figure** |
| `pfeaturesInterpretation/{krr,gpr}_feature_directionality.pdf` | NEW supplementary |
| `pfeaturesInterpretation/{linear,lasso}_feature_coefficients.pdf` | NEW supplementary |

**Leaky vs clean R² for the canonical Ridge headline figure** (target = Rg norm w/0.406, Inliers, No regr out, 15 obs across 5 seeds × 3 splits): leaky Ridge = +0.020, clean Ridge = +0.031 (Δ +0.011 — within noise). The headline figure is qualitatively unchanged; only the CSV-level scaler leakage is removed.

Strongest configuration anywhere in the sweep: **Kernel Ridge on the un-normalised Rg target, Inliers, no regr out, R²=0.653** (not the target the original figures used). Separate experimental-design question from the leakage fix.

## Stage 6 — Side artifacts (CLI logs / status)

| File | Description |
|------|-------------|
| `esmScripts/sweep.log` | stdout+stderr from `esmRegressionsAllSeeds.py` background run |
| `OUTPUTS_MANIFEST.md` | this file |
