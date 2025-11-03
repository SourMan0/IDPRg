import numpy as np
import pandas as pd
import csv
import re
from pathlib import Path
from doAllRegressions import evaluate_models_rmse

# ======================
# Load label sequences
# ======================

all_points_df = pd.read_csv('training/all_points.csv')
inliers_df = pd.read_csv('training/inliers.csv')

all_sequences = all_points_df['Sequence'].to_list()
inlier_sequences = inliers_df['Sequence'].to_list()

# ======================
# Locate all PCA embedding files
# ======================
data_path = Path("data")
pca_files = sorted(data_path.glob("protbert_*_PCA*.csv"))

if not pca_files:
    raise FileNotFoundError("No PCA embedding files found in ./data/. Expected files like protBert_low_PCA10.csv")

print(f"Found {len(pca_files)} PCA embedding files:")
for f in pca_files:
    print(" -", f.name)

# ======================
# Load label data
# ======================
labelHeaders = [
    'Sequence', 'Rg (nm)', 'Rg normalized w/0.427', 'Rg normalized w/0.5 (nm)', 'Rg normalized w/0.418 (nm)',
    'Rg w/pH regressed out', 'Rg normalized w/0.427 w/pH regressed out', 'Rg normalized w/0.5 w/pH regressed out',
    'Rg normalized w/0.418 w/pH regressed out', 'Rg w/buffer regressed out', 'Rg normalized w/0.427 w/buffer regressed out',
    'Rg normalized w/0.5 w/buffer regressed out', 'Rg normalized w/0.418 w/buffer regressed out',
    'Rg w/experimental pH regressed out', 'Rg normalized w/0.427 w/experimental pH regressed out',
    'Rg normalized w/0.5 w/experimental pH regressed out', 'Rg normalized w/0.418 w/experimental pH regressed out',
    'Rg w/experimental buffer regressed out', 'Rg normalized w/0.427 w/experimental buffer regressed out',
    'Rg normalized w/0.5 w/experimental buffer regressed out', 'Rg normalized w/0.418 w/experimental buffer regressed out'
]

labels = [[] for _ in range(len(labelHeaders) - 1)]
inlierLabels = [[] for _ in range(len(labelHeaders) - 1)]

with open('training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        for i, val in enumerate(row[1:]):
            labels[i].append(float(val))

with open('training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        for i, val in enumerate(row[1:]):
            inlierLabels[i].append(float(val))

labels = np.array(labels, dtype=float)
inlierLabels = np.array(inlierLabels, dtype=float)

# ======================
# Prepare inlier mask (remove outliers)
# ======================
outlierIndices = [123, 136, 151, 158, 171, 185]
inl = np.ones(190, dtype=bool)
inl[outlierIndices] = False

# ======================
# Label splits for descriptive output
# ======================
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
            seq_col = 'Sequence' if 'Sequence' in df.columns else 'Experimental Sequence'
            df.columns = [seq_col] + [f'pc{i+1}' for i in range(df.shape[1] - 1)]

            df_all = df[df[seq_col].isin(all_sequences)].copy()
            df_inl = df[df[seq_col].isin(inlier_sequences)].copy()

            X_all = df_all.drop(columns=[seq_col]).to_numpy()
            X_inl = df_inl.drop(columns=[seq_col]).to_numpy()

            print(f"Running PCA{pca_num} | {layer_group}: {X_all.shape[0]} all, {X_inl.shape[0]} inliers")

            for label_idx, ls in enumerate(labelSplits):
                y_all = labels[label_idx]
                y_inl = inlierLabels[label_idx]

                losses_all = evaluate_models_rmse(X_all, y_all, seed)
                losses_inl = evaluate_models_rmse(X_inl, y_inl, seed)

                for loss in losses_all:
                    writer.writerow([ls[0], ls[1], 'All', 'ProtBERT', layer_group, pca_num,
                                     loss[0], loss[1], loss[2], loss[3]])
                for loss in losses_inl:
                    writer.writerow([ls[0], ls[1], 'Inliers', 'ProtBERT', layer_group, pca_num,
                                     loss[0], loss[1], loss[2], loss[3]])

print("✅ All ProtBERT PCA regressions completed. Results saved to protbertLosses_allLayers_seed*.csv")