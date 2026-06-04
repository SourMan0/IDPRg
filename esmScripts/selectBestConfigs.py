"""Find the best (ESM mode, layer, PCA, model) configurations from the
leak-free sweep CSVs (esmLosses1..5.csv in cwd).

Usage:
    python selectBestConfigs.py [--top 20] [--target "Rg norm w/0.5"]
                                [--points Inliers] [--regr-out "No regr out"]

Aggregates across the 5 seeds and the 3 train/test splits by mean R^2.
Prints the top configs and writes best_configs.csv summary.
"""

import argparse
import csv
import glob
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--target', default=None,
                    help='Filter by Normalization column value (e.g. "Rg norm w/0.5"). '
                         'If omitted, all targets are mixed.')
    ap.add_argument('--regr-out', default=None,
                    help='Filter by "Regressing out" column (e.g. "No regr out").')
    ap.add_argument('--points', default=None,
                    help='Filter by "Points" column ("All" or "Inliers").')
    ap.add_argument('--model', default=None,
                    help='Filter by model (e.g. "Lasso", "Kernel Ridge", "GPR").')
    ap.add_argument('--csv-glob', default='esmLosses*.csv')
    ap.add_argument('--out', default='best_configs.csv')
    args = ap.parse_args()

    files = sorted(glob.glob(args.csv_glob))
    # Exclude any *_leaky.csv that might be in the same dir
    files = [f for f in files if 'leaky' not in f.lower()]
    if not files:
        raise SystemExit(f'No CSVs matched {args.csv_glob}')
    print(f'Reading {len(files)} CSV(s): {files}')

    # bucket key -> list of R^2
    buckets = defaultdict(list)
    n_rows = 0
    for fp in files:
        with open(fp, newline='') as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                n_rows += 1
                norm, regr, pts, esm, layer, pca, model, split, r2, rmse = row
                if args.target and norm != args.target:
                    continue
                if args.regr_out and regr != args.regr_out:
                    continue
                if args.points and pts != args.points:
                    continue
                if args.model and model != args.model:
                    continue
                key = (norm, regr, pts, esm, layer, pca, model)
                buckets[key].append(float(r2))
    print(f'Read {n_rows} rows, {len(buckets)} configurations after filtering')

    summary = []
    for key, r2s in buckets.items():
        mean_r2 = sum(r2s) / len(r2s)
        summary.append((mean_r2, len(r2s)) + key)
    summary.sort(reverse=True, key=lambda t: t[0])

    print(f'\nTop {args.top} configs by mean R^2 (across seeds × splits):')
    print(f'{"mean_R2":>8} {"n":>3}  {"Target":<22} {"Regr":<28} '
          f'{"Points":<8} {"ESM":<7} {"Lyr":<3} {"PCA":<5} Model')
    for row in summary[:args.top]:
        mr, n, norm, regr, pts, esm, layer, pca, model = row
        print(f'{mr:8.4f} {n:3d}  {norm:<22} {regr:<28} '
              f'{pts:<8} {esm:<7} {layer:<3} {pca:<5} {model}')

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mean_R2', 'n_obs', 'Normalization', 'Regressing out',
                    'Points', 'ESM Mode', 'Layer', 'Principal Components',
                    'Regression Type'])
        for row in summary:
            w.writerow(row)
    print(f'\nFull ranking -> {args.out}')


if __name__ == '__main__':
    main()
