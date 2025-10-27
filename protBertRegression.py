import numpy as np
import pandas as pd
import csv
from doAllRegressions import evaluate_models_rmse

# ======================
# Load label sequences
# ======================

all_points_df = pd.read_csv('training/all_points.csv')
inliers_df = pd.read_csv('training/inliers.csv')

all_sequences = all_points_df['Sequence'].to_list()
inlier_sequences = inliers_df['Sequence'].to_list()

# ======================
# Load ProtBERT PCA embeddings
# ======================

PCAvals = [10, 20, 50, 100, 190]
protbert_features = {}          # features for all sequences
protbert_inlier_features = {}   # features for inliers only

for p in PCAvals:
    # Load PCA embeddings
    df = pd.read_csv(f'data/protbertEmbeddings2PCA{p}2.csv')
    df = df.iloc[:, :-1]  # drop last column (old Rg)
    df.columns = ['sequence'] + [f'pc{i+1}' for i in range(df.shape[1] - 1)]

    # ----------------------
    # Filter embeddings to match all_points.csv
    # ----------------------
    df_all = df[df['sequence'].isin(all_sequences)].copy()
    X_all = df_all.drop(columns=['sequence']).to_numpy()
    protbert_features[p] = X_all

    # ----------------------
    # Filter embeddings to match inliers.csv
    # ----------------------
    df_inl = df[df['sequence'].isin(inlier_sequences)].copy()
    X_inl = df_inl.drop(columns=['sequence']).to_numpy()
    protbert_inlier_features[p] = X_inl

    # Print for sanity check
    print(f'Loaded PCA{p}: {X_all.shape[0]} sequences (all), {X_inl.shape[0]} sequences (inliers)')

# ======================
# Load Rg label data (same as ESM version)
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
# Prepare inlier features (remove outliers)
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
# Run regressions and save results for multiple seeds
# ======================

for idx, seed in enumerate([43, 44, 45, 46]):
    outfile = f'protbertLosses{idx+1}.csv'
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Normalization', 'Regressing out', 'Points', 'Model', 'PCA Components', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
        writer.writerow(header)

        for label_idx, ls in enumerate(labelSplits):
            y_all = labels[label_idx]
            y_inl = inlierLabels[label_idx]

            for p in PCAvals:
                X = protbert_features[p]
                Xi = protbert_inlier_features[p]

                losses_all = evaluate_models_rmse(X, y_all, seed)
                losses_inl = evaluate_models_rmse(Xi, y_inl, seed)

                for loss in losses_all:
                    writer.writerow([ls[0], ls[1], 'All', 'ProtBERT', p, loss[0], loss[1], loss[2], loss[3]])

                for loss in losses_inl:
                    writer.writerow([ls[0], ls[1], 'Inliers', 'ProtBERT', p, loss[0], loss[1], loss[2], loss[3]])

print("ProtBERT regression runs completed. Results saved to protbertLosses*.csv")