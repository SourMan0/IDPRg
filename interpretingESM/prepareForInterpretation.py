# LEAK-FREE rewrite of prepareForInterpretation.py
# ------------------------------------------------
# Previously this script fit a free-standing StandardScaler+PCA on all 163
# inliers, *then* ran GridSearchCV(cv=5) over Lasso / KernelRidge alpha & gamma.
# Even though no test set was being held out (the resulting model is used only
# for interpretation, not test reporting), the CV alpha/gamma selection saw
# PCA-leaky features because PCA had already pooled covariance across all
# inner-train folds.
#
# The fix is one line of concept: wrap StandardScaler+PCA+model in a sklearn
# Pipeline and pass *that* to GridSearchCV. PCA is then refit on each inner
# train fold during alpha selection. The final pipeline is refit on all 163
# inliers (refit=True by default) -- that's the artifact we dump for the
# sliding-window occlusion downstream.
#
# Outputs (both leak-free, pickled as full Pipelines so callers no longer need
# to load PCA + model separately):
#   lasso_pipeline.joblib  -- Pipeline([StandardScaler, PCA, Lasso])
#   krr_pipeline.joblib    -- Pipeline([StandardScaler, PCA, KernelRidge])
#
# Configs default to the previously selected (layer, PCA dim) picks so that
# the sliding-window outputs can be regenerated immediately.  Once the
# leak-free sweep finishes (sweep.log -> esmLosses{1..5}.csv) we revisit
# these picks; rerun this script with --lasso-* / --krr-* overrides if the
# new clean R^2 ranking points elsewhere.

import argparse
import csv

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTLIER_INDICES = [114, 125, 137, 163]


def load_inliers_X(pt_path: str, layer: int) -> np.ndarray:
    """Return mean-pooled?  No -- the joblib is consumed downstream by
    sliding-window scripts that call `pipeline.transform(mean_pooled_vec)` and
    `pipeline.predict(...)`. To be consistent with that, we train the pipeline
    on per-sequence *mean-pooled* layer embeddings, exactly mirroring how the
    occlusion code (tryingDifferentType2.py::predict_rg_from_embedding) feeds
    the model at interpretation time.

    The OLD script used the raw (167, hidden_dim) tensor sliced by layer, which
    is already the mean of the (L, hidden_dim) per-residue embeddings produced
    upstream when ESM was run with mean pooling. Same thing -- just be
    explicit.
    """
    t = torch.load(pt_path)
    X_np = np.asarray(t.detach().cpu(), dtype=np.float64)
    inl = np.ones(X_np.shape[0], dtype=bool)
    inl[OUTLIER_INDICES] = False
    X_layer = X_np[:, layer, :]
    return X_layer[inl]


def load_inlier_targets(inliers_csv: str, target_col: int) -> np.ndarray:
    y = []
    with open(inliers_csv, newline='') as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            y.append(float(row[target_col]))
    return np.asarray(y, dtype=float)


def fit_lasso_pipeline(X: np.ndarray, y: np.ndarray, n_components: int,
                       cv: int = 5, random_state: int = 42):
    """Leak-free α selection: StandardScaler+PCA refit per CV fold."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=random_state)),
        ('model', Lasso(max_iter=10000)),
    ])
    grid = {'model__alpha': np.logspace(-4, 1, 10)}
    gs = GridSearchCV(pipe, grid, cv=cv, scoring='r2', refit=True)
    gs.fit(X, y)
    print(f'  Best Lasso α: {gs.best_params_["model__alpha"]:.4g}  '
          f'(CV R² = {gs.best_score_:.4f})')
    return gs.best_estimator_


def fit_krr_pipeline(X: np.ndarray, y: np.ndarray, n_components: int,
                     cv: int = 5, random_state: int = 42):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=random_state)),
        ('model', KernelRidge(kernel='rbf')),
    ])
    grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10],
        'model__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1],
    }
    gs = GridSearchCV(pipe, grid, cv=cv, scoring='r2', refit=True)
    gs.fit(X, y)
    print(f'  Best KRR α={gs.best_params_["model__alpha"]}, '
          f'γ={gs.best_params_["model__gamma"]}  '
          f'(CV R² = {gs.best_score_:.4f})')
    return gs.best_estimator_


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lasso-esm', choices=['esm6', 'esm12'], default='esm12',
                    help='ESM model for the Lasso pipeline (default: esm12)')
    ap.add_argument('--lasso-layer', type=int, default=6,
                    help='Layer index for the Lasso pipeline (default: 6)')
    ap.add_argument('--lasso-pca', type=int, default=50,
                    help='PCA dim for the Lasso pipeline (default: 50)')
    ap.add_argument('--krr-esm', choices=['esm6', 'esm12'], default='esm6',
                    help='ESM model for the KRR pipeline (default: esm6)')
    ap.add_argument('--krr-layer', type=int, default=3,
                    help='Layer index for the KRR pipeline (default: 3)')
    ap.add_argument('--krr-pca', type=int, default=100,
                    help='PCA dim for the KRR pipeline (default: 100)')
    ap.add_argument('--target-col', type=int, default=3,
                    help='Column index in inliers.csv to predict (default: 3 = "Rg normalized w/0.5")')
    ap.add_argument('--inliers-csv', default='../training/inliers.csv')
    ap.add_argument('--embed-dir', default='../esmScripts/esm_embeddings')
    args = ap.parse_args()

    pt_paths = {
        'esm6':  f'{args.embed_dir}/esm6layer.pt',
        'esm12': f'{args.embed_dir}/esm12layer.pt',
    }

    y = load_inlier_targets(args.inliers_csv, args.target_col)

    # ---- Lasso pipeline ---------------------------------------------------
    print(f'>>> Fitting Lasso pipeline on {args.lasso_esm} layer '
          f'{args.lasso_layer} (PCA={args.lasso_pca}) ...')
    X_lasso = load_inliers_X(pt_paths[args.lasso_esm], args.lasso_layer)
    print(f'    X shape: {X_lasso.shape}')
    lasso_pipe = fit_lasso_pipeline(X_lasso, y, n_components=args.lasso_pca)
    joblib.dump(lasso_pipe, 'lasso_pipeline.joblib')
    print('    -> lasso_pipeline.joblib')

    # ---- KRR pipeline -----------------------------------------------------
    print(f'>>> Fitting KRR pipeline on {args.krr_esm} layer '
          f'{args.krr_layer} (PCA={args.krr_pca}) ...')
    X_krr = load_inliers_X(pt_paths[args.krr_esm], args.krr_layer)
    print(f'    X shape: {X_krr.shape}')
    krr_pipe = fit_krr_pipeline(X_krr, y, n_components=args.krr_pca)
    joblib.dump(krr_pipe, 'krr_pipeline.joblib')
    print('    -> krr_pipeline.joblib')


if __name__ == '__main__':
    main()
