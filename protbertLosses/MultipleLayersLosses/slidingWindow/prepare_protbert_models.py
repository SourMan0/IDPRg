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

    if reg_type_lower in ("kernel_ridge", "kernelridge", "kr", "kernel", "kernal ridge"):
        return KernelRidge(kernel="rbf", alpha=1.0, gamma=None)

    if reg_type_lower in ("rf", "randomforest", "random_forest"):
        return RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )

    if reg_type_lower in ("gpr", "gaussianprocess", "gaussian_process"):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel()
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=seed,
        )

    raise ValueError(f"Unknown reg_type '{reg_type}'")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _find_and_rename_seq_col(emb_df: pd.DataFrame, desired_seq_col: str) -> pd.DataFrame:
    """
    Make embeddings DF have the same sequence column name as train_csv by renaming
    from common alternatives if needed.
    """
    emb_df = emb_df.copy()

    if desired_seq_col in emb_df.columns:
        return emb_df

    candidates = [
        "Sequence",
        "Experimental Sequence",
        "Protein Sequence",
        "protein_sequence",
        "seq",
        "Seq",
    ]
    for c in candidates:
        if c in emb_df.columns:
            print(f"  Renaming embeddings sequence column '{c}' -> '{desired_seq_col}'")
            return emb_df.rename(columns={c: desired_seq_col})

    raise ValueError(
        f"Could not find a sequence column in embeddings CSV.\n"
        f"Expected '{desired_seq_col}' or one of {candidates}.\n"
        f"Columns found: {list(emb_df.columns)}"
    )


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
    train_df = _standardize_columns(pd.read_csv(args.train_csv))

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

    # Keep only what we need from training CSV (prevents accidental numeric leakage)
    train_df = train_df[[args.sequence_col, args.target_col]].copy()
    train_df[args.target_col] = pd.to_numeric(train_df[args.target_col], errors="coerce")
    train_df = train_df.dropna(subset=[args.target_col])

    # Load selected model configs
    with open(args.configs_json, "r") as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        raise ValueError("configs_json should be a list of model configs.")

    if args.top_k is not None:
        configs = configs[: args.top_k]

    print(f"Loaded {len(configs)} configs (using top_k = {args.top_k}).")

    for i, cfg in enumerate(configs, start=1):
        layer_group = get_cfg_field(cfg, "layer_group", "LayerGroup", "group", required=True)
        pca_dim = int(get_cfg_field(cfg, "pca_dim", "PCA", "pca", "PCA Components", required=True))
        reg_type = get_cfg_field(cfg, "reg_type", "RegType", "regressor", "Regression Type", default="Ridge")
        seed = int(get_cfg_field(cfg, "seed", "Seed", "random_state", default=0))
        layer_idx = int(get_cfg_field(cfg, "layer_idx", "layer", "LayerIndex", default=15))

        print(f"\n[model_{i}] group={layer_group}, layer_idx={layer_idx}, PCA={pca_dim}, reg={reg_type}, seed={seed}")

        # 1) Load raw embeddings for this group
        emb_fname = args.layer_pattern.format(group=layer_group)
        emb_path = os.path.join(args.emb_dir, emb_fname)
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Could not find raw embedding CSV at {emb_path}.\n"
                f"Either place the file there or adjust --layer_pattern."
            )

        emb_df = _standardize_columns(pd.read_csv(emb_path))
        print(f"  Loaded embeddings from {emb_path} with shape {emb_df.shape}")

        # 2) Ensure embeddings have the same sequence_col name as train_df
        emb_df = _find_and_rename_seq_col(emb_df, args.sequence_col)

        # 3) Merge by sequence (NEVER assume row order)
        merged = train_df.merge(emb_df, on=args.sequence_col, how="inner")

        if merged.empty:
            raise ValueError(
                f"No overlapping sequences between train_csv and embeddings ({emb_fname}).\n"
                f"Check that sequences are identical strings and same column is used."
            )

        if len(merged) < len(train_df):
            print(
                f"  [info] Using intersection only: {len(merged)} matched sequences "
                f"out of {len(train_df)} labeled training sequences."
            )

        # 4) Build X_raw from numeric columns in embeddings (exclude target + seq)
        num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        # exclude target if it’s numeric
        num_cols = [c for c in num_cols if c != args.target_col]

        if not num_cols:
            raise ValueError(
                f"No numeric embedding columns found after merge for {emb_fname}.\n"
                f"Columns: {list(merged.columns)}"
            )

        X_raw = merged[num_cols].to_numpy(dtype=float)
        y = merged[args.target_col].to_numpy(dtype=float)

        print(f"  Using X_raw.shape={X_raw.shape}, y.shape={y.shape}")

        # 5) Fit PCA in RAW embedding space
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_pca = pca.fit_transform(X_raw)
        print(f"  PCA fitted: X_pca.shape = {X_pca.shape}")

        # 6) Fit regressor on PCA features
        reg = build_regressor(reg_type, seed)
        reg.fit(X_pca, y)

        preds = reg.predict(X_pca)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"  Train RMSE on this model: {rmse:.4f}")

        # 7) Save artifacts
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
                n_train_rows=len(train_df),
                n_matched_rows=len(merged),
                embedding_num_cols=len(num_cols),
            )
        )
        with open(cfg_path, "w") as f:
            json.dump(cfg_out, f, indent=2)

        print(f"  Saved PCA → {pca_path}")
        print(f"  Saved regressor → {reg_path}")
        print(f"  Saved config → {cfg_path}")


if __name__ == "__main__":
    main()
