import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import csv  # you actually don't need this anymore, but kept for consistency
from doAllRegressions import evaluate_models_rmse

# ======================
# Make sure output directory exists
# ======================
os.makedirs("protbertLosses/MultipleLayersLosses", exist_ok=True)

# ======================
# Load label sequences & dataframes
# ======================

all_points_df = pd.read_csv('training/all_points.csv')
inliers_df = pd.read_csv('training/inliers.csv')

all_sequences = all_points_df['Sequence'].to_list()
inlier_sequences = inliers_df['Sequence'].to_list()

# ======================
# Locate all PCA embedding files
# ======================
data_path = Path("data/protbert_embeddings")
pca_files = sorted(data_path.glob("protbert_*_PCA*.csv"))

if not pca_files:
    raise FileNotFoundError("No PCA embedding files found in ./data/protbert_embeddings. "
                            "Expected files like protbert_low_PCA10.csv")

print(f"Found {len(pca_files)} PCA embedding files:")
for f in pca_files:
    print(" -", f.name)

# ======================
# Label header names (columns in training CSVs)
# ======================
labelHeaders = [
    'Sequence',
    'Rg (nm)',
    'Rg normalized w/0.427',
    'Rg normalized w/0.5 (nm)',
    'Rg normalized w/0.418 (nm)',
    'Rg w/pH regressed out',
    'Rg normalized w/0.427 w/pH regressed out',
    'Rg normalized w/0.5 w/pH regressed out',
    'Rg normalized w/0.418 w/pH regressed out',
    'Rg w/buffer regressed out',
    'Rg normalized w/0.427 w/buffer regressed out',
    'Rg normalized w/0.5 w/buffer regressed out',
    'Rg normalized w/0.418 w/buffer regressed out',
    'Rg w/experimental pH regressed out',
    'Rg normalized w/0.427 w/experimental pH regressed out',
    'Rg normalized w/0.5 w/experimental pH regressed out',
    'Rg normalized w/0.418 w/experimental pH regressed out',
    'Rg w/experimental buffer regressed out',
    'Rg normalized w/0.427 w/experimental buffer regressed out',
    'Rg normalized w/0.5 w/experimental buffer regressed out',
    'Rg normalized w/0.418 w/experimental buffer regressed out'
]

# Human-readable descriptions for the 20 label variants (index 0..19)
# label_idx i --> uses column labelHeaders[i+1]
labelSplits = [
    ['Rg w/no norm', 'No regr out'],
    ['Rg norm w/0.427', 'No regr out'],
    ['Rg norm w/0.5', 'No regr out'],
    ['Rg norm w/0.418', 'No regr out'],
    ['Rg w/no norm', 'pH regr out'],
    ['Rg norm w/0.427', 'pH regr out'],
    ['Rg norm w/0.5', 'pH regr out'],
    ['Rg norm w/0.418', 'pH regr out'],
    ['Rg w/no norm', 'buffer regr out'],
    ['Rg norm w/0.427', 'buffer regr out'],
    ['Rg norm w/0.5', 'buffer regr out'],
    ['Rg norm w/0.418', 'buffer regr out'],
    ['Rg w/no norm', 'expr pH only regr out'],
    ['Rg norm w/0.427', 'expr pH only regr out'],
    ['Rg norm w/0.5', 'expr pH only regr out'],
    ['Rg norm w/0.418', 'expr pH only regr out'],
    ['Rg w/no norm', 'expr buffer only regr out'],
    ['Rg norm w/0.427', 'expr buffer only regr out'],
    ['Rg norm w/0.5', 'expr buffer only regr out'],
    ['Rg norm w/0.418', 'expr buffer only regr out']
]

# ======================
# Run regressions across all PCA files
# ======================
for idx, seed in enumerate([43, 44, 45, 46]):
    outfile = f'protbertLosses/MultipleLayersLosses/protbertLosses{idx+1}.csv'
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'Normalization', 'Regressing out', 'Points', 'Model', 'LayerGroup',
            'PCA Components', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score'
        ]
        writer.writerow(header)

        for file in pca_files:
            filename = file.name

            # Extract layer group and PCA component count from filename
            layer_match = re.search(r"protbert_(low|mid|high)", filename)
            pca_match = re.search(r"PCA(\d+)", filename)

            if not layer_match or not pca_match:
                print(f"⚠️ Skipping file (unrecognized pattern): {filename}")
                continue

            layer_group = layer_match.group(1)
            pca_num = int(pca_match.group(1))

            # Load embeddings
            df = pd.read_csv(file)

            # Figure out which column is the sequence column
            if 'Sequence' in df.columns:
                seq_col = 'Sequence'
            elif 'Experimental Sequence' in df.columns:
                seq_col = 'Experimental Sequence'
            else:
                raise KeyError(f"No sequence column found in {filename}")

            # Rename PCA columns uniformly: pc1, pc2, ...
            df.columns = [seq_col] + [f'pc{i+1}' for i in range(df.shape[1] - 1)]

            # Filter to sequences present in all_points / inliers
            df_all = df[df[seq_col].isin(all_sequences)].copy()
            df_inl = df[df[seq_col].isin(inlier_sequences)].copy()

            X_cols = [c for c in df_all.columns if c.startswith('pc')]
            print(f"Running PCA{pca_num} | {layer_group}: {df_all.shape[0]} all, {df_inl.shape[0]} inliers")

            for label_idx, ls in enumerate(labelSplits):
                # label_idx 0..19 -> column labelHeaders[label_idx+1]
                target_col = labelHeaders[label_idx + 1]

                # ===== All points =====
                merged_all = df_all.merge(
                    all_points_df[['Sequence', target_col]],
                    left_on=seq_col,
                    right_on='Sequence',
                    how='inner'
                )

                X_all = merged_all[X_cols].to_numpy()
                y_all = merged_all[target_col].to_numpy()

                # ===== Inliers =====
                merged_inl = df_inl.merge(
                    inliers_df[['Sequence', target_col]],
                    left_on=seq_col,
                    right_on='Sequence',
                    how='inner'
                )

                X_inl = merged_inl[X_cols].to_numpy()
                y_inl = merged_inl[target_col].to_numpy()

                # Sanity check (optional)
                # print(label_idx, target_col, X_all.shape, y_all.shape, X_inl.shape, y_inl.shape)

                losses_all = evaluate_models_rmse(X_all, y_all, seed)
                losses_inl = evaluate_models_rmse(X_inl, y_inl, seed)

                for loss in losses_all:
                    writer.writerow([
                        ls[0], ls[1], 'All', 'ProtBERT', layer_group, pca_num,
                        loss[0], loss[1], loss[2], loss[3]
                    ])

                for loss in losses_inl:
                    writer.writerow([
                        ls[0], ls[1], 'Inliers', 'ProtBERT', layer_group, pca_num,
                        loss[0], loss[1], loss[2], loss[3]
                    ])

print(" All ProtBERT PCA regressions completed. Results saved to protbertLosses/MultipleLayersLosses/protbertLosses*.csv")