# LEAK-FREE rewrite of the physical-features sweep.
#
# Previously this script read protein_features3St2.csv which had been
# StandardScaler.fit_transform'd across all 167 sequences by
# extractFeaturesFromPoints3St.py before the train/test split ever happened.
# That leaked test-set mean/std into the supposedly-held-out evaluation.
#
# extractFeaturesFromPoints3St.py now writes raw features. Here we pass
# do_scaling=True so evaluate_models_rmse wraps StandardScaler+model in a
# Pipeline that is refit on every training fold.
#
# Output schema is unchanged (8 columns) so downstream consumers keep working.

import csv
import numpy as np

from doAllRegressions import evaluate_models_rmse


def main():
    features1 = []
    with open('protein_features3St2.csv', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            features1.append(row[1:])
    features1 = np.array(features1, dtype=float)
    print(f'features1 shape: {features1.shape}')

    labelHeaders = [
        'Sequence', 'Rg (nm)', 'Rg normalized w/0.421', 'Rg normalized w/0.5 (nm)',
        'Rg normalized w/0.406 (nm)',
        'Rg w/pH regressed out', 'Rg normalized w/0.421 w/pH regressed out',
        'Rg normalized w/0.5 w/pH regressed out',
        'Rg normalized w/0.406 w/pH regressed out',
        'Rg w/buffer regressed out', 'Rg normalized w/0.421 w/buffer regressed out',
        'Rg normalized w/0.5 w/buffer regressed out',
        'Rg normalized w/0.406 w/buffer regressed out',
        'Rg w/experimental pH regressed out',
        'Rg normalized w/0.421 w/experimental pH regressed out',
        'Rg normalized w/0.5 w/experimental pH regressed out',
        'Rg normalized w/0.406 w/experimental pH regressed out',
        'Rg w/experimental buffer regressed out',
        'Rg normalized w/0.421 w/experimental buffer regressed out',
        'Rg normalized w/0.5 w/experimental buffer regressed out',
        'Rg normalized w/0.406 w/experimental buffer regressed out',
    ]
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
    labels = [np.asarray(c, dtype=float) for c in labels]
    inlierLabels = [np.asarray(c, dtype=float) for c in inlierLabels]

    outlierIndices = [114, 125, 137, 163]
    inl = np.ones(167, dtype=bool)
    inl[outlierIndices] = False
    inlierFeatures1 = features1[inl]

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

    SEEDS = [42, 43, 44, 45, 46]
    header = ['Normalization', 'Regressing out', 'Points', 'Feature Selection',
              'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']

    for idx, seed in enumerate(SEEDS, start=1):
        out_path = f'pfeatureLosses{idx}St3.csv'
        print(f'>>> seed {seed} -> {out_path}', flush=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for li, ls in enumerate(labelSplits):
                losses_all = evaluate_models_rmse(features1, labels[li],
                                                  random_state=seed, do_scaling=True)
                losses_inl = evaluate_models_rmse(inlierFeatures1, inlierLabels[li],
                                                  random_state=seed, do_scaling=True)
                for r_all, r_inl in zip(losses_all, losses_inl):
                    writer.writerow([ls[0], ls[1], 'All',     '1',
                                     r_all[0], r_all[1], r_all[2], r_all[3]])
                    writer.writerow([ls[0], ls[1], 'Inliers', '1',
                                     r_inl[0], r_inl[1], r_inl[2], r_inl[3]])


if __name__ == '__main__':
    main()
