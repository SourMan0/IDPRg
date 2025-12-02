#!/usr/bin/env python3
# prepare_protbert_models.py
#
# Build PCA + regressor for your top ProtBERT models, in the raw embedding space.

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.kernel_ridge import KernelRidge



def get_cfg_field(cfg, *names, default=None, required=False):
    """
    Helper to be tolerant to key naming in selected_models.json.
    E.g. allow 'layer_group' or 'LayerGroup', 'PCA Components', etc.
    """
    for n in names:
        if n in cfg:
            return cfg[n]
    if required:
        raise KeyError(f"None of {names} found in config: {cfg}")
    return default


def build_regressor(reg_type, seed):
    reg_type_lower = str(reg_type).lower()

    if reg_type_lower == "ridge":
        return Ridge(random_state=seed)

    if reg_type_lower == "lasso":
        return Lasso(random_state=seed)

    if reg_type_lower in ("kernel_ridge", "kernelridge", "kr", "kernel", 'kernal ridge'):
        # You can tune kernel, alpha, gamma here
        return KernelRidge(
            kernel="rbf",   # radial basis function kernel
            alpha=1.0,      # regularization
            gamma=None      # auto-gamma if None
        )

    if reg_type_lower in ("rf", "randomforest", "random_forest"):
        return RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )

    if reg_type_lower in ("gpr", "gaussianprocess", "gaussian_process"):
        # Simple but reasonable kernel for your PCA features
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel()
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=seed,
        )

    raise ValueError(f"Unknown reg_type '{reg_type}'")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs_json", required=True,
                    help="JSON file with selected top ProtBERT models.")
    ap.add_argument("--train_csv", required=True,
                    help="CSV with training points and target Rg.")
    ap.add_argument("--sequence_col", required=True,
                    help="Column name for protein sequences in train_csv.")
    ap.add_argument("--target_col", required=True,
                    help="Column name for target Rg.")
    ap.add_argument("--emb_dir", required=True,
                    help="Directory containing raw ProtBERT embedding CSVs.")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write model_1/, model_2/, ... subdirs.")
    ap.add_argument("--top_k", type=int, default=None,
                    help="If set, only build first k configs from JSON.")
    ap.add_argument(
        "--layer_pattern",
        default="protbert_{group}.csv",
        help=(
            "Filename pattern (inside emb_dir) for raw embeddings.\n"
            "Use '{group}' as placeholder for the layer group (e.g. low/mid/high).\n"
            "Example: 'protbert_{group}.csv' if your files are named that way."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load training table
    train_df = pd.read_csv(args.train_csv)
    if args.sequence_col not in train_df.columns:
        raise ValueError(
            f"sequence_col='{args.sequence_col}' not found in train_csv columns.\n"
            f"Columns: {list(train_df.columns)}"
        )
    if args.target_col not in train_df.columns:
        raise ValueError(
            f"target_col='{args.target_col}' not found in train_csv columns.\n"
            f"Columns: {list(train_df.columns)}"
        )

    y_full = train_df[args.target_col].to_numpy(dtype=float)

    # Load selected model configs
    with open(args.configs_json, "r") as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        raise ValueError("configs_json should be a list of model configs.")

    if args.top_k is not None:
        configs = configs[: args.top_k]

    print(f"Loaded {len(configs)} configs (using top_k = {args.top_k}).")

    for i, cfg in enumerate(configs, start=1):
        # Tolerant to your key names:
        # {'LayerGroup': 'mid', 'PCA Components': 100, 'Regression Type': 'Lasso', 'Seed': 3, ...}
        layer_group = get_cfg_field(cfg, "layer_group", "LayerGroup", "group",
                                    required=True)
        pca_dim = int(
            get_cfg_field(
                cfg,
                "pca_dim",
                "PCA",
                "pca",
                "PCA Components",  # <- your key
                required=True,
            )
        )
        reg_type = get_cfg_field(
            cfg,
            "reg_type",
            "RegType",
            "regressor",
            "Regression Type",  # <- your key
            default="Ridge",
        )
        seed = int(get_cfg_field(cfg, "seed", "Seed", "random_state", default=0))
        layer_idx = int(get_cfg_field(cfg, "layer_idx", "layer", "LayerIndex", default=15))

        print(
            f"\n[model_{i}] group={layer_group}, layer_idx={layer_idx}, "
            f"PCA={pca_dim}, reg={reg_type}, seed={seed}"
        )

        # 1) Load raw embeddings for this group (1024-dim)
        # Build filename based on layer group + PCA dimension
        emb_fname = args.layer_pattern.format(group=layer_group)
        emb_path = os.path.join(args.emb_dir, emb_fname)
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Could not find raw embedding CSV at {emb_path}.\n"
                f"Either place the file there or adjust --layer_pattern."
            )
        emb_df = pd.read_csv(emb_path)
        print(f"  Loaded embeddings from {emb_path} with shape {emb_df.shape}")

        # --- Handle sequence column name mismatch: 'Sequence' vs 'Protein Sequence' ---
        if args.sequence_col not in emb_df.columns and "Sequence" in emb_df.columns:
            print(
                f"  Renaming 'Sequence' column in embeddings to '{args.sequence_col}' "
                "to match train_csv."
            )
            emb_df = emb_df.rename(columns={"Sequence": args.sequence_col})

        # Try to align embeddings with train_df via sequence_col if present;
        # otherwise assume same row order.
        if args.sequence_col in emb_df.columns:
            merged = train_df[[args.sequence_col, args.target_col]].merge(
                emb_df,
                on=args.sequence_col,
                how="inner",
            )
            if len(merged) != len(train_df):
                print(
                    f"  [warn] merge on '{args.sequence_col}' yielded "
                    f"{len(merged)} rows vs {len(train_df)} in train_csv. "
                    f"Using intersection only."
                )
            num_cols = merged.select_dtypes(include=[np.number]).columns
            num_cols = [c for c in num_cols if c != args.target_col]
            X_raw = merged[num_cols].to_numpy(dtype=float)
            y = merged[args.target_col].to_numpy(dtype=float)
        else:
            # No sequence column in embedding CSV -> assume same order
            num_cols = emb_df.select_dtypes(include=[np.number]).columns
            X_raw = emb_df[num_cols].to_numpy(dtype=float)
            y = y_full
            if X_raw.shape[0] != len(y):
                raise ValueError(
                    f"Row mismatch: embeddings have {X_raw.shape[0]} rows, "
                    f"but train_csv has {len(y)} targets."
                )

        print(f"  Using raw embedding matrix X_raw.shape = {X_raw.shape}")

        # 2) Fit PCA in RAW embedding space
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_pca = pca.fit_transform(X_raw)
        print(f"  PCA fitted: X_pca.shape = {X_pca.shape}")

        # 3) Fit regressor on PCA features
        reg = build_regressor(reg_type, seed)
        reg.fit(X_pca, y)
        preds = reg.predict(X_pca)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"  Train RMSE on this model: {rmse:.4f}")

        # 4) Save to model_i directory
        model_dir = os.path.join(args.out_dir, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)

        pca_path = os.path.join(model_dir, "pca.joblib")
        reg_path = os.path.join(model_dir, "regressor.joblib")
        cfg_path = os.path.join(model_dir, "config.json")

        joblib.dump(pca, pca_path)
        joblib.dump(reg, reg_path)

        cfg_out = dict(cfg)
        cfg_out.update(
            dict(
                layer_group=layer_group,
                layer_idx=layer_idx,
                pca_dim=pca_dim,
                reg_type=reg_type,
                seed=seed,
                target_col=args.target_col,
                sequence_col=args.sequence_col,
                embeddings_csv=emb_fname,
            )
        )
        with open(cfg_path, "w") as f:
            json.dump(cfg_out, f, indent=2)

        print(f"  Saved PCA → {pca_path}")
        print(f"  Saved regressor → {reg_path}")
        print(f"  Saved config → {cfg_path}")


if __name__ == "__main__":
    main()
