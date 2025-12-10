# Best model: ['Rg norm w/0.418' 'No regr out' 'All' '10' 'Kernel Ridge' '90/10']

import csv
import ast
import numpy as np
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from jax_unirep import get_reps
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def get_unirep_per_res(seq: str) -> np.ndarray:
    """
    Approximate per-residue UniRep embeddings by using the final hidden state
    on all prefixes of the sequence.

    Parameters
    ----------
    seq : str
        Protein sequence.

    Returns
    -------
    per_res : np.ndarray, shape (L, 1900)
        Row t is the (approximate) representation for position t
        based on UniRep's hidden state after seeing residues 0..t.
    """
    L = len(seq)
    per_res = np.zeros((L, 1900), dtype=np.float32)

    for t in range(L):
        prefix = seq[:t+1]
        # get_reps returns (h_avg, h_final, c_final) each of shape (1, 1900)
        _, h_final, _ = get_reps(prefix)
        per_res[t, :] = h_final[0]

    return per_res


def main():
    # 1. Load sequences and targets
    sequences = []
    y = []

    with open('training/all_points.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            seq = row[0]
            target = float(row[4])  # adjust index if your target column is different
            sequences.append(seq)
            y.append(target)

    y = np.array(y, dtype=float)

    # 2. Compute UniRep h_avg for each sequence
    X_list = []
    df_X = pd.read_csv("data/unirep_allRaw.csv", converters={"UniRep Embedding": ast.literal_eval})
    X_list = df_X["UniRep Embedding"].tolist()
    # for i, seq in enumerate(sequences):
    #     per_res = get_unirep_per_res(seq)  # (L, 1900)
    #     h_avg = per_res.mean(axis=0)  # (1900,)
    #     X_list.append(h_avg)

    #     if True:
    #         print(f"Computed UniRep embeddings for {i+1} sequences")

    X_raw = np.stack(X_list, axis=0)  # (N, 1900)

    # 3. PCA
    pca = PCA(n_components=10, random_state=42)
    X = pca.fit_transform(X_raw)
    joblib.dump(pca, "unirep_pca.joblib")
    print("Saved PCA to unirep_pca.joblib")

    krr_alphas = [0.001, 0.01, 0.1, 1, 10]
    krr_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    model = GridSearchCV(
        KernelRidge(kernel="rbf"),
        {"alpha": krr_alphas, "gamma": krr_gammas},
        cv=5, scoring="r2", n_jobs=-1
    )

    print("Fitting model...")
    model.fit(X, y)
    joblib.dump(model, "unirep_krr.joblib")
    print("Saved KRR to unirep_krr.joblib")


if __name__ == "__main__":
    main()
