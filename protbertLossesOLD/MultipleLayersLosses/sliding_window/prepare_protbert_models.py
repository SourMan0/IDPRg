#!/usr/bin/env python3
# prepare_protbert_models.py

import argparse, os, json, glob
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split

def load_embeddings(emb_path):
    if os.path.isdir(emb_path):
        files = sorted(glob.glob(os.path.join(emb_path, "*.pt")) +
                       glob.glob(os.path.join(emb_path, "*.npy")))
        if not files:
            raise FileNotFoundError(f"No .pt/.npy files in {emb_path}")
        arrs = []
        for f in files:
            if f.endswith(".pt"):
                import torch
                x = torch.load(f, map_location="cpu")
                x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.array(x)
            else:
                x = np.load(f)
            if x.ndim == 2:  # per-residue -> mean pool
                x = x.mean(axis=0)
            arrs.append(x)
        return np.stack(arrs, axis=0)

    if emb_path.endswith(".pt"):
        import torch
        X = torch.load(emb_path, map_location="cpu")
        X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.array(X)
    else:
        X = np.load(emb_path)

    if X.ndim == 3:  # (N,L,H) -> mean pool L
        X = X.mean(axis=1)
    if X.ndim != 2:
        raise ValueError(f"Embeddings must be (N,H) or (N,L,H). Got {X.shape}")
    return X

def build_regressor(kind, seed):
    k = kind.strip().lower()
    if k in ["kernel ridge", "krr", "kernelridge"]:
        return KernelRidge(alpha=1.0, kernel="rbf", gamma=None)
    if k in ["gpr", "gaussian process", "gaussianprocess"]:
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-5)
        return GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True)
    raise ValueError(f"Unknown Regression Type: {kind}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top5_csv", required=True)
    ap.add_argument("--train_csv", default="training/all_points.csv")
    ap.add_argument("--sequence_col", default="Sequence")
    ap.add_argument("--target_col", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out_dir", default="protbert_top_models_all_only")
    ap.add_argument("--test_size", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    top5 = pd.read_csv(args.top5_csv)
    if "Points" not in top5.columns:
        raise KeyError(f"'Points' column missing in {args.top5_csv}")

    top5_all = top5[top5["Points"].astype(str).str.strip().str.lower() == "all"].copy()
    if len(top5_all) == 0:
        raise ValueError("No models left after filtering Points=='All'.")

    print(f"Keeping {len(top5_all)} top models trained on ALL points.")

    train_df = pd.read_csv(args.train_csv)
    if args.target_col not in train_df.columns:
        raise KeyError(f"{args.target_col} not in {args.train_csv}")

    y = train_df[args.target_col].values.astype(float)
    X = load_embeddings(args.embeddings)

    if len(X) != len(train_df):
        raise ValueError(f"Embeddings N={len(X)} != sequences N={len(train_df)}")

    for rank, (_, row) in enumerate(top5_all.iterrows(), start=1):
        n_pca = int(row["PCA Components"])
        reg_type = row["Regression Type"]
        seed = int(row.get("Seed", 1))

        pca = PCA(n_components=n_pca, random_state=seed)
        X_pca = pca.fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_pca, y, test_size=args.test_size, random_state=seed
        )

        reg = build_regressor(reg_type, seed)
        reg.fit(X_tr, y_tr)

        model_dir = os.path.join(args.out_dir, f"model_{rank}")
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(pca, os.path.join(model_dir, "pca.joblib"))
        joblib.dump(reg, os.path.join(model_dir, "regressor.joblib"))

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({
                "rank": rank,
                "pca_components": n_pca,
                "regression_type": reg_type,
                "seed": seed,
                "points": row["Points"],
                "model": row.get("Model", None),
            }, f, indent=2)

        print(f"[saved] model_{rank}: PCA={n_pca}, reg={reg_type}")

if __name__ == "__main__":
    main()