#Move to parent directory to remake csvs
#
# LEAK-FREE, PARALLELISED REWRITE
# -------------------------------
# Previously this script loaded pre-baked PCA features from 6esmPCA/ and 12esmPCA/,
# where StandardScaler+PCA had been fit on all 167 sequences before any train/test
# split (see esmScripts/6esmPCAMake.py + esmScripts/12esmPCAMake.py).  That leaked
# test-point covariance into both the scaler and PCA basis.
#
# Now we load the raw per-layer embeddings (esm_embeddings/*.pt) and pass them to
# doAllRegressions.evaluate_models_rmse with n_components=d.  Inside that function
# StandardScaler+PCA+model live in a sklearn Pipeline that is only fit on X_train
# (and only on the inner-CV train folds during GridSearchCV) so the test point
# never contributes to any preprocessor.
#
# We parallelise the outer (label, esm_mode, layer, PCA, points-subset) loop
# across a ProcessPoolExecutor so the 5-seed full sweep fits in a single overnight
# run on a 12-core laptop instead of ~3 days.  Each worker pins BLAS / OpenMP to a
# single thread so the workers don't fight, and the raw embedding arrays are
# loaded once per worker via the pool initializer instead of being re-pickled on
# every submit.
#
# Output: esmLosses{1..5}.csv with the same 10-column schema as before; the
# 'Principal Components' column now records the *actual* PCA dim used (==d unless
# d > 0.8*n_train, in which case it's capped so PCA fits inside every CV fold).

import os
# Pin BLAS / OpenMP threads so worker processes don't fight each other.
# MUST be set before numpy / sklearn are imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from doAllRegressions import evaluate_models_rmse


PCAvals = [167, 100, 50, 20, 10]
outlierIndices = [114, 125, 137, 163]
inl = np.ones(167, dtype=bool)
inl[outlierIndices] = False

SEEDS = [42, 43, 44, 45, 46]

labelHeaders = [
    'Sequence',
    'Rg (nm)', 'Rg normalized w/0.421', 'Rg normalized w/0.5 (nm)', 'Rg normalized w/0.406 (nm)',
    'Rg w/pH regressed out', 'Rg normalized w/0.421 w/pH regressed out',
    'Rg normalized w/0.5 w/pH regressed out', 'Rg normalized w/0.406 w/pH regressed out',
    'Rg w/buffer regressed out', 'Rg normalized w/0.421 w/buffer regressed out',
    'Rg normalized w/0.5 w/buffer regressed out', 'Rg normalized w/0.406 w/buffer regressed out',
    'Rg w/experimental pH regressed out',
    'Rg normalized w/0.421 w/experimental pH regressed out',
    'Rg normalized w/0.5 w/experimental pH regressed out',
    'Rg normalized w/0.406 w/experimental pH regressed out',
    'Rg w/experimental buffer regressed out',
    'Rg normalized w/0.421 w/experimental buffer regressed out',
    'Rg normalized w/0.5 w/experimental buffer regressed out',
    'Rg normalized w/0.406 w/experimental buffer regressed out',
]

labelSplits = [
    ['Rg w/no norm', 'No regr out'],
    ['Rg norm w/0.421', 'No regr out'],
    ['Rg norm w/0.5', 'No regr out'],
    ['Rg norm w/0.406', 'No regr out'],
    ['Rg w/no norm', 'pH regr out'],
    ['Rg norm w/0.421', 'pH regr out'],
    ['Rg norm w/0.5', 'pH regr out'],
    ['Rg norm w/0.406', 'pH regr out'],
    ['Rg w/no norm', 'buffer regr out'],
    ['Rg norm w/0.421', 'buffer regr out'],
    ['Rg norm w/0.5', 'buffer regr out'],
    ['Rg norm w/0.406', 'buffer regr out'],
    ['Rg w/no norm', 'expr pH only regr out'],
    ['Rg norm w/0.421', 'expr pH only regr out'],
    ['Rg norm w/0.5', 'expr pH only regr out'],
    ['Rg norm w/0.406', 'expr pH only regr out'],
    ['Rg w/no norm', 'expr buffer only regr out'],
    ['Rg norm w/0.421', 'expr buffer only regr out'],
    ['Rg norm w/0.5', 'expr buffer only regr out'],
    ['Rg norm w/0.406', 'expr buffer only regr out'],
]

HEADER = ['Normalization', 'Regressing out', 'Points', 'ESM Mode', 'Layer',
          'Principal Components', 'Regression Type', 'Test Split',
          'Test R2 Score', 'RMSE Score']


# ---------------------------------------------------------------------------
# Worker globals (populated once per worker by the pool initializer)
# ---------------------------------------------------------------------------

_X = {}   # (esm_mode, points, layer) -> (n_samples, n_features) array
_Y = {}   # (points, label_idx) -> (n_samples,) array


def _init_worker(X_payload, Y_payload):
    global _X, _Y
    _X = X_payload
    _Y = Y_payload


def _worker(task):
    seed, li, ls0, ls1, esm_mode, layer, d, points = task
    X = _X[(esm_mode, points, layer)]
    y = _Y[(points, li)]
    rows = evaluate_models_rmse(X, y, random_state=seed, n_components=d)
    out = []
    for model_name, split_name, r2, rmse, d_used in rows:
        out.append([ls0, ls1, points, esm_mode, layer, d_used,
                    model_name, split_name, r2, rmse])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_raw(path):
    t = torch.load(path)
    return np.asarray(t.detach().cpu(), dtype=np.float64)


def main():
    print('Loading raw ESM embeddings...', flush=True)
    X6 = load_raw('esm_embeddings/esm6layer.pt')
    X12 = load_raw('esm_embeddings/esm12layer.pt')
    n_layers_6 = X6.shape[1]
    n_layers_12 = X12.shape[1]
    print(f'  ESM-6: {X6.shape}, ESM-12: {X12.shape}', flush=True)

    X_payload = {}
    for layer in range(n_layers_6):
        X_payload[('ESM-6', 'All', layer)] = X6[:, layer, :]
        X_payload[('ESM-6', 'Inliers', layer)] = X6[inl][:, layer, :]
    for layer in range(n_layers_12):
        X_payload[('ESM-12', 'All', layer)] = X12[:, layer, :]
        X_payload[('ESM-12', 'Inliers', layer)] = X12[inl][:, layer, :]

    n_targets = len(labelHeaders) - 1
    labels = [[] for _ in range(n_targets)]
    inlierLabels = [[] for _ in range(n_targets)]
    with open('../training/all_points.csv', newline='') as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            for c, val in enumerate(row[1:]):
                labels[c].append(val)
    with open('../training/inliers.csv', newline='') as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            for c, val in enumerate(row[1:]):
                inlierLabels[c].append(val)
    Y_payload = {}
    for li in range(n_targets):
        Y_payload[('All', li)] = np.asarray(labels[li], dtype=float)
        Y_payload[('Inliers', li)] = np.asarray(inlierLabels[li], dtype=float)

    n_workers = max(1, min(os.cpu_count() or 1, 12))
    print(f'Using {n_workers} worker processes', flush=True)

    ctx = mp.get_context('spawn')

    for idx, seed in enumerate(SEEDS, start=1):
        out_path = f'esmLosses{idx}.csv'
        print(f'\n>>> Seed {seed} -> {out_path}', flush=True)
        t_start = time.time()

        tasks = []
        for li, ls in enumerate(labelSplits):
            for layer in range(n_layers_6):
                for d in PCAvals:
                    tasks.append((seed, li, ls[0], ls[1], 'ESM-6', layer, d, 'All'))
                    tasks.append((seed, li, ls[0], ls[1], 'ESM-6', layer, d, 'Inliers'))
            for layer in range(n_layers_12):
                for d in PCAvals:
                    tasks.append((seed, li, ls[0], ls[1], 'ESM-12', layer, d, 'All'))
                    tasks.append((seed, li, ls[0], ls[1], 'ESM-12', layer, d, 'Inliers'))
        print(f'  {len(tasks)} tasks queued', flush=True)

        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            done = 0
            with ProcessPoolExecutor(max_workers=n_workers,
                                     mp_context=ctx,
                                     initializer=_init_worker,
                                     initargs=(X_payload, Y_payload)) as ex:
                # imap_unordered-style: submit then drain in completion order
                futures = [ex.submit(_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    rows = fut.result()
                    for r in rows:
                        writer.writerow(r)
                    done += 1
                    if done % 200 == 0 or done == len(tasks):
                        elapsed = time.time() - t_start
                        rate = done / elapsed if elapsed > 0 else 0.0
                        eta = (len(tasks) - done) / rate if rate > 0 else float('inf')
                        print(f'  {done}/{len(tasks)} tasks done '
                              f'({elapsed/60:.1f} min elapsed, ETA {eta/60:.1f} min)',
                              flush=True)
                        f.flush()
        elapsed = time.time() - t_start
        print(f'<<< Seed {seed} done in {elapsed/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
