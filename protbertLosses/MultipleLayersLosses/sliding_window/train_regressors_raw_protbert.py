#!/usr/bin/env python3
# train_regressors_raw_protbert.py
"""
Train frozen PCA + regressor DIRECTLY from raw ProtBERT pooled embeddings.

Inputs:
- all_models.json (configs from extract_all_models.py)
- all_points.csv (Sequence + Rg target)
Outputs:
- out_dir/model_1/{pca.joblib, regressor.joblib, config.json}
...

This fixes feature-dimension mismatch for sliding-window.
"""

import argparse, os, json, pickle
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split

PROTBERT_NAME = "Rostlab/prot_bert"

# If your low/mid/high definition differs, edit these.
LAYER_MAP = {"low": 6, "mid": 15, "high": 24}


def build_regressor(kind, seed):
    k = kind.strip().lower()
    if k in ["kernel ridge", "krr", "kernelridge"]:
        return KernelRidge(alpha=1.0, kernel="rbf")
    if k in ["gpr", "gaussian process", "gaussianprocess"]:
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-5)
        return GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True)
    raise ValueError(f"Unknown Regression Type: {kind}")


def get_layer_embeddings(seq, tokenizer, model, layer_idx, device="cpu"):
    spaced = " ".join(list(seq))
    toks = tokenizer(spaced, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**toks, output_hidden_states=True)
        h = out.hidden_states[layer_idx][0]  # (T, H)
    return h[1:-1].cpu().numpy()  # strip CLS/SEP


def pooled_vec(seq, tokenizer, model, layer_idx, device="cpu"):
    res_emb = get_layer_embeddings(seq, tokenizer, model, layer_idx, device)
    return res_emb.mean(axis=0)  # (1024,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_json", required=True)
    ap.add_argument("--targets_csv", required=True)
    ap.add_argument("--sequence_col", default="Sequence")
    ap.add_argument("--target_col", default="Rg normalized w/0.5 (nm)")
    ap.add_argument("--out_dir", default="protbert_top_models_all_only_raw")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache_embeddings", action="store_true",
                    help="Cache pooled embeddings per layer_group to disk for speed.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    configs = json.load(open(args.models_json))

    df = pd.read_csv(args.targets_csv)
    if args.sequence_col not in df.columns:
        raise KeyError(f"{args.sequence_col} not found in {args.targets_csv}")
    if args.target_col not in df.columns:
        rg_candidates = [c for c in df.columns if "rg" in c.lower()]
        if not rg_candidates:
            raise KeyError(f"{args.target_col} not found and no Rg-like columns exist.")
        args.target_col = rg_candidates[0]
        print(f"[auto] using target_col='{args.target_col}'")

    seqs = df[args.sequence_col].astype(str).tolist()
    y_full = df[args.target_col].astype(float).values

    tokenizer = AutoTokenizer.from_pretrained(PROTBERT_NAME, do_lower_case=False)
    model = AutoModel.from_pretrained(PROTBERT_NAME).to(args.device)
    model.eval()

    # Precompute pooled embeddings for each layer group needed (low/mid/high)
    layer_groups_needed = sorted(set(c["LayerGroup"] for c in configs))
    pooled_by_group = {}

    for lg in layer_groups_needed:
        lg = str(lg).lower()
        layer_idx = LAYER_MAP[lg]

        cache_path = os.path.join(args.out_dir, f"pooled_{lg}.npy")
        if args.cache_embeddings and os.path.exists(cache_path):
            X = np.load(cache_path)
            pooled_by_group[lg] = X
            print(f"[cache hit] loaded pooled embeddings for {lg} from {cache_path}")
            continue

        print(f"[compute] pooling ProtBERT embeddings for layer_group={lg} (layer={layer_idx})")
        X_list = []
        for seq in tqdm(seqs, desc=f"pool {lg}"):
            X_list.append(pooled_vec(seq, tokenizer, model, layer_idx, args.device))
        X = np.stack(X_list, axis=0)  # (N, 1024)
        pooled_by_group[lg] = X

        if args.cache_embeddings:
            np.save(cache_path, X)
            print(f"[cache save] {cache_path}")

    # Train each top model with its own PCA size + regressor type
    for rank, cfg in enumerate(configs, start=1):
        lg = str(cfg["LayerGroup"]).lower()
        n_pca = int(cfg["PCA Components"])
        reg_type = str(cfg["Regression Type"])
        seed = int(cfg.get("Seed", 1))

        X_raw = pooled_by_group[lg]

        # Fit PCA from raw pooled 1024-d vectors
        pca = PCA(n_components=n_pca, random_state=seed)
        X_pca = pca.fit_transform(X_raw)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_pca, y_full, test_size=args.test_size, random_state=seed
        )

        reg = build_regressor(reg_type, seed)
        reg.fit(X_tr, y_tr)

        mdir = os.path.join(args.out_dir, f"model_{rank}")
        os.makedirs(mdir, exist_ok=True)

        joblib.dump(pca, os.path.join(mdir, "pca.joblib"))
        joblib.dump(reg, os.path.join(mdir, "regressor.joblib"))

        with open(os.path.join(mdir, "config.json"), "w") as f:
            json.dump({
                **cfg,
                "layer_idx_used": LAYER_MAP[lg],
                "trained_from": "raw ProtBERT pooled embeddings",
                "target_col_used": args.target_col
            }, f, indent=2)

        print(f"[saved] model_{rank}: group={lg}, PCA={n_pca}, reg={reg_type}")

if __name__ == "__main__":
    main()