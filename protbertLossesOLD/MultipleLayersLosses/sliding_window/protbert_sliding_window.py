#!/usr/bin/env python3
# protbert_sliding_window.py

import argparse, os
import numpy as np
import joblib
import torch
import json
from transformers import AutoTokenizer, AutoModel

PROTBERT_NAME = "Rostlab/prot_bert"

PROTBERT_NAME = "Rostlab/prot_bert"

LAYER_MAP = {
    "low": 6,
    "mid": 15,
    "high": 24
}

class IdentityTransform:
    def transform(self, X):
        return X


def load_frozen(model_dir):
    pca_path = os.path.join(model_dir, "pca.joblib")
    reg_path = os.path.join(model_dir, "regressor.joblib")
    cfg_path = os.path.join(model_dir, "config.json")

    # PCA is identity anyway; if loading fails, fall back safely
    try:
        pca = joblib.load(pca_path)
    except Exception as e:
        print(f"[warn] Could not load {pca_path}. Using IdentityTransform instead.")
        print("       error was:", repr(e))
        pca = IdentityTransform()

    reg = joblib.load(reg_path)
    cfg = json.load(open(cfg_path))

    return pca, reg, cfg


def get_layer_embeddings(seq, tokenizer, model, layer_idx=4, device="cpu"):
    spaced = " ".join(list(seq))
    toks = tokenizer(spaced, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**toks, output_hidden_states=True)
        h = out.hidden_states[layer_idx][0]  # (T, H)
    return h[1:-1].cpu().numpy()  # strip [CLS], [SEP]

def predict_from_embeddings(res_emb, pca, reg):
    feat = res_emb.mean(axis=0, keepdims=True)  # mean pool
    feat_pca = pca.transform(feat)
    return float(reg.predict(feat_pca)[0])

def token_mask_effect(seq, pca, reg, tokenizer, model, layer_idx=4, window=5, device="cpu"):
    L = len(seq)
    base_emb = get_layer_embeddings(seq, tokenizer, model, layer_idx, device)
    baseline = predict_from_embeddings(base_emb, pca, reg)

    effects = np.zeros(L, dtype=float)
    windows, frags = [], []

    for start in range(L - window + 1):
        end = start + window
        masked = list(seq)
        masked[start:end] = ["X"] * window   # simple AA-space masking
        masked_seq = "".join(masked)

        masked_emb = get_layer_embeddings(masked_seq, tokenizer, model, layer_idx, device)
        masked_pred = predict_from_embeddings(masked_emb, pca, reg)

        delta = baseline - masked_pred
        center = start + window // 2
        effects[center] = delta

        windows.append((start, end))
        frags.append(seq[start:end])

    return baseline, effects, windows, frags

def embedding_occlusion_effect(seq, pca, reg, tokenizer, model, layer_idx=4, window=5, mode="zero", device="cpu"):
    res_emb = get_layer_embeddings(seq, tokenizer, model, layer_idx, device)
    baseline = predict_from_embeddings(res_emb, pca, reg)

    L = res_emb.shape[0]
    effects = np.zeros(L, dtype=float)
    windows, frags = [], []
    mean_vec = res_emb.mean(axis=0, keepdims=True)

    for start in range(L - window + 1):
        end = start + window
        occ = res_emb.copy()
        if mode == "zero":
            occ[start:end, :] = 0.0
        elif mode == "mean":
            occ[start:end, :] = mean_vec
        else:
            raise ValueError("mode must be zero or mean")

        occ_pred = predict_from_embeddings(occ, pca, reg)

        delta = baseline - occ_pred
        center = start + window // 2
        effects[center] = delta

        windows.append((start, end))
        frags.append(seq[start:end])

    return baseline, effects, windows, frags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--layer_idx", type=int, default=4)
    ap.add_argument("--method", choices=["token", "embed"], default="embed")
    ap.add_argument("--occlusion_mode", choices=["zero", "mean"], default="zero")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(PROTBERT_NAME, do_lower_case=False)
    model = AutoModel.from_pretrained(PROTBERT_NAME).to(args.device)
    model.eval()

    pca, reg = load_frozen(args.model_dir)

    if args.method == "token":
        baseline, effects, windows, frags = token_mask_effect(
            args.seq, pca, reg, tokenizer, model,
            layer_idx=args.layer_idx, window=args.window, device=args.device
        )
    else:
        baseline, effects, windows, frags = embedding_occlusion_effect(
            args.seq, pca, reg, tokenizer, model,
            layer_idx=args.layer_idx, window=args.window,
            mode=args.occlusion_mode, device=args.device
        )

    print("baseline:", baseline)
    print("effects:", effects)

if __name__ == "__main__":
    main()