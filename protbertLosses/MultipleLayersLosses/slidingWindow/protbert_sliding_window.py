#!/usr/bin/env python3
# protbert_sliding_window.py
"""
Sliding-window occlusion on ProtBERT embeddings for Rg prediction.

Exports
-------
- PROTBERT_NAME : HuggingFace model name
- LAYER_MAP     : mapping {'low','mid','high'} -> layer index
- load_frozen   : load PCA + regressor + config from a model_* directory
- embedding_occlusion_effect : run occlusion on a single sequence
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from joblib import load as joblib_load

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

# Change this if you used a different ProtBERT
PROTBERT_NAME = "Rostlab/prot_bert"

# Map your LayerGroup string to a hidden layer index
# (these are typical choices; adjust if your training used different layers)
LAYER_MAP = {
    "low": 5,
    "mid": 15,
    "high": 23,
}


# ---------------------------------------------------------------------
# Helper: prediction from embeddings
# ---------------------------------------------------------------------

def predict_from_embeddings(emb, pca, reg):
    """
    emb : numpy array of shape (1, 1024) or (n, 1024)
    pca : fitted sklearn PCA (or similar) with .transform()
    reg : fitted sklearn regressor with .predict()

    Returns:
      float prediction for the first row.
    """
    # Ensure 2D
    if emb.ndim == 1:
        emb = emb[None, :]

    # Apply PCA if provided
    if pca is not None:
        feat_pca = pca.transform(emb)   # -> (1, n_components)
    else:
        feat_pca = emb

    pred = reg.predict(feat_pca)       # -> (1,)
    return float(pred[0])


# ---------------------------------------------------------------------
# Loading frozen PCA + regressor + config
# ---------------------------------------------------------------------

import os
import json
import joblib

def load_frozen(model_dir):
    """
    Load PCA + regressor + config from a frozen ProtBERT model directory.

    Expected files:
      - pca.joblib
      - regressor.joblib
      - config.json
    """
    pca_path = os.path.join(model_dir, "pca.joblib")
    reg_path = os.path.join(model_dir, "regressor.joblib")
    cfg_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA not found at {pca_path}")

    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Regressor not found at {reg_path}")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found at {cfg_path}")

    pca = joblib.load(pca_path)
    reg = joblib.load(reg_path)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    return pca, reg, cfg



# ---------------------------------------------------------------------
# Sliding-window occlusion
# ---------------------------------------------------------------------

def _sequence_to_tokens(seq):
    """
    ProtBERT expects amino acids separated by spaces, e.g.:
      'A E D K ...'
    """
    return " ".join(list(seq))


def embedding_occlusion_effect(
    seq,
    pca,
    reg,
    tokenizer,
    model,
    layer_idx,
    window=5,
    mode="zero",
    device="cpu",
):
    """
    Perform sliding-window occlusion on a single sequence.

    Parameters
    ----------
    seq : str
        Amino-acid sequence (no spaces).
    pca, reg :
        Frozen PCA + regressor as loaded by load_frozen().
    tokenizer, model :
        ProtBERT tokenizer & model.
    layer_idx : int
        Which hidden layer to use (e.g. 5, 15, 23).
    window : int
        Window size (in residues).
    mode : {"zero","mean"}
        How to occlude:
          - "zero": set hidden vectors in the window to 0
          - "mean": set them to the mean embedding across the sequence
    device : str
        "cpu" or "cuda"

    Returns
    -------
    baseline_pred : float
        Model prediction on the unoccluded sequence.
    effects : np.ndarray, shape (num_windows,)
        Effect of occluding each window (occluded - baseline).
    windows : list of (start, end) tuples
        Index range in [0, len(seq)) for each window.
    frags : list of str
        Sequence fragment for each window.
    """
    model.eval()

    # -----------------------------
    # Tokenize and forward once
    # -----------------------------
    toks = _sequence_to_tokens(seq)
    inputs = tokenizer(
        toks,
        return_tensors="pt",
        add_special_tokens=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple length n_layers+1 (including embedding layer)
    hidden_states = outputs.hidden_states
    layer_emb = hidden_states[layer_idx]  # (1, L, hidden_dim)

    # Drop [CLS] and [SEP] to align with residues
    # Assuming ProtBERT gives [CLS] + L_res + [SEP]
    token_embs = layer_emb[0, 1:-1, :]  # (L_res, hidden_dim)
    L = token_embs.shape[0]
    hidden_dim = token_embs.shape[1]

    if L != len(seq):
        # This shouldn't happen for standard ProtBERT AA tokenization,
        # but we guard just in case.
        # We'll truncate to min length.
        minL = min(L, len(seq))
        token_embs = token_embs[:minL, :]
        seq = seq[:minL]
        L = minL

    # -----------------------------
    # Baseline embedding & prediction
    # -----------------------------
    # Simple scheme: mean over residue positions
    baseline_emb = token_embs.mean(dim=0, keepdim=True)  # (1, hidden_dim)
    baseline_emb_np = baseline_emb.cpu().numpy()
    baseline_pred = predict_from_embeddings(baseline_emb_np, pca, reg)

    # -----------------------------
    # Sliding windows
    # -----------------------------
    if L < window:
        # Too short for this window; return no windows
        return baseline_pred, np.array([]), [], []

    token_embs_np = token_embs.cpu().numpy()  # (L, hidden_dim)
    effects = []
    windows = []
    frags = []

    # Precompute mean if we need it
    global_mean = token_embs_np.mean(axis=0, keepdims=True)

    for start in range(0, L - window + 1):
        end = start + window

        occluded = token_embs_np.copy()
        if mode == "zero":
            occluded[start:end, :] = 0.0
        elif mode == "mean":
            occluded[start:end, :] = global_mean
        else:
            raise ValueError(f"Unknown occlusion mode: {mode}")

        # Pool again (mean)
        occl_emb = occluded.mean(axis=0, keepdims=True)  # (1, hidden_dim)
        occl_pred = predict_from_embeddings(occl_emb, pca, reg)

        effect = float(occl_pred - baseline_pred)

        effects.append(effect)
        windows.append((start, end))
        frags.append(seq[start:end])

    effects = np.array(effects, dtype=float)
    return baseline_pred, effects, windows, frags
