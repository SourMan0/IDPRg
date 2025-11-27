#!/usr/bin/env python3
# train_regressors_from_pca_csv.py

import argparse, os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split

class IdentityTransform:
    def transform(self, X): return X

def build_regressor(kind, seed):
    k = kind.strip().lower()
    if k in ["kernel ridge", "krr", "kernelridge"]:
        return KernelRidge(alpha=1.0, kernel="rbf")
    if k in ["gpr", "gaussian process", "gaussianprocess"]:
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-5)
        return GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True)
    raise ValueError(f"Unknown Regression Type: {kind}")

def clean_seq(s):
    # robust matching
    return str(s).strip().upper().replace(" ", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_json", required=True)
    ap.add_argument("--pca_dir", required=True)
    ap.add_argument("--targets_csv", required=True,
                    help="CSV with Sequence + Rg targets (your all_points.csv).")
    ap.add_argument("--sequence_col", default="Sequence")
    ap.add_argument("--target_col", default="Rg normalized w/0.5 (nm)")
    ap.add_argument("--out_dir", default="protbert_top_models_all_only")
    ap.add_argument("--test_size", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = json.load(open(args.models_json))

    # ---- load targets ----
    targets_df = pd.read_csv(args.targets_csv)
    if args.sequence_col not in targets_df.columns:
        raise KeyError(f"{args.sequence_col} not in {args.targets_csv}")

    # If target_col missing, try auto-find an Rg-like numeric column
    if args.target_col not in targets_df.columns:
        rg_candidates = [c for c in targets_df.columns if "rg" in c.lower()]
        if not rg_candidates:
            raise KeyError(f"{args.target_col} not in {args.targets_csv}, and no 'Rg' column found.")
        args.target_col = rg_candidates[0]
        print(f"[auto] using target_col='{args.target_col}'")

    targets_df["_SEQKEY_"] = targets_df[args.sequence_col].map(clean_seq)
    targets_map = dict(zip(targets_df["_SEQKEY_"], targets_df[args.target_col].astype(float)))

    for rank, cfg in enumerate(configs, start=1):
        layer = cfg["LayerGroup"]
        n_pca = cfg["PCA Components"]
        reg_type = cfg["Regression Type"]
        seed = cfg.get("Seed", 1)

        pca_csv = os.path.join(args.pca_dir, f"protbert_{layer}_PCA{n_pca}.csv")
        if not os.path.exists(pca_csv):
            raise FileNotFoundError(f"Missing PCA feature file: {pca_csv}")

        df = pd.read_csv(pca_csv)
        if args.sequence_col not in df.columns:
            raise KeyError(f"{args.sequence_col} not in {pca_csv}")

        df["_SEQKEY_"] = df[args.sequence_col].map(clean_seq)

        # attach y by matching sequences
        df["y"] = df["_SEQKEY_"].map(targets_map)
        before = len(df)
        df = df.dropna(subset=["y"]).copy()
        after = len(df)

        if after == 0:
            raise ValueError(f"No sequences in {pca_csv} matched targets in {args.targets_csv}.")
        if after < before:
            print(f"[warn] {before-after} sequences in {pca_csv} had no target match; dropped.")

        y = df["y"].values.astype(float)

        # numeric PCA columns as X
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["y"]]
        X = df[feature_cols].values.astype(float)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=seed)
        reg = build_regressor(reg_type, seed)
        reg.fit(X_tr, y_tr)

        model_dir = os.path.join(args.out_dir, f"model_{rank}")
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(IdentityTransform(), os.path.join(model_dir, "pca.joblib"))
        joblib.dump(reg, os.path.join(model_dir, "regressor.joblib"))

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({
                **cfg,
                "pca_csv_used": pca_csv,
                "targets_csv_used": args.targets_csv,
                "target_col_used": args.target_col,
                "feature_cols_used": feature_cols,
                "matched_rows": after
            }, f, indent=2)

        print(f"[saved] model_{rank}: layer={layer}, PCA={n_pca}, reg={reg_type}, matched={after}")

if __name__ == "__main__":
    main()