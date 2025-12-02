#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protein_cip.py — Sliding-mask occlusion + Continuance Importance Profile (CIP)

What it does
------------
Given a CSV with a 'Sequence' column (protein sequences), this script:
  1) Computes per-position occlusion importances using a sliding mask window
  2) Builds Continuance Importance Profiles (CIP) by cumulative summation
  3) Aggregates mean±std curves across aligned positions
  4) Saves CSV outputs and two PNG plots

How to use (examples)
---------------------
# (A) With YOUR OWN model: implement grammar_predict() below, then:
python protein_cip.py \
  --points_csv all_points.csv \
  --target_col "Rg normalized w/0.5 (nm)" \
  --window 5 \
  --strategy mask \
  --out_dir results_cip

# (B) With a surrogate (Ridge on AA composition) for a quick dry run:
python protein_cip.py \
  --points_csv all_points.csv \
  --target_col "Rg normalized w/0.5 (nm)" \
  --window 5 \
  --strategy mask \
  --out_dir results_cip \
  --use_surrogate

Outputs
-------
- <out_dir>/occlusion_importance.csv
- <out_dir>/continuance_profiles.csv
- <out_dir>/summary_importance.csv
- <out_dir>/mean_importance.png
- <out_dir>/mean_cip.png

Equations
---------
Windowed occlusion score (window width w, start i):
    I_i = f(x) - f(x_{[i:i+w]<-mask})

Per-position importance (equal share of each window, averaged over overlaps):
    per_pos[j] = (1 / cnt[j]) * sum_{i: j in [i, i+w)} ( I_i / w )

Normalized per-position importance:
    per_pos_norm[j] = per_pos[j] / (sum_{k=0..L-1} |per_pos[k]| + eps)

Continuance Importance Profile (CIP):
    CIP[k] = sum_{j=0..k} per_pos[j]

Normalized CIP:
    CIP_norm[k] = (sum_{j=0..k} per_pos[j]) / (sum_{t=0..L-1} |per_pos[t]| + eps)
"""

import os
import argparse
from typing import Optional, Callable, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional (only needed for the surrogate model)
try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -------------------------------
# 1) Plug YOUR model here
# -------------------------------
def grammar_predict(seq: str) -> float:
    """
    TODO: Replace this function with your protein grammar model's inference.
    It must return a single scalar (float) prediction for the sequence `seq`.
    Example stub:
        return float(your_model.predict(seq))
    """
    raise NotImplementedError("Implement grammar_predict() or run with --use_surrogate.")


# -------------------------------
# 2) Surrogate (for quick dry runs)
# -------------------------------
def build_surrogate_predictor(df: pd.DataFrame, seed: int = 0) -> Callable[[str], float]:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn required for --use_surrogate. Install scikit-learn or implement grammar_predict().")

    AMINO_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ALPHABET)}

    def featurize(seq: str) -> np.ndarray:
        counts = np.zeros(len(AMINO_ALPHABET), dtype=float)
        for ch in seq:
            counts[aa_to_idx.get(ch, aa_to_idx["X"])] += 1.0
        if len(seq) > 0:
            counts /= len(seq)
        return counts

    X = np.vstack([featurize(s) for s in df["Sequence"]])
    y = df["__target__"].astype(float).values

    model = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeCV(alphas=np.logspace(-6, 6, 25), cv=None)
    )
    model.fit(X, y)

    def predict_fn(seq: str) -> float:
        return float(model.predict(featurize(seq)[None, :])[0])

    return predict_fn


# -------------------------------
# 3) Masking + occlusion utilities
# -------------------------------
MASK_TOKEN = "X"

def mask_window(seq: str, start: int, width: int, strategy: str = "mask") -> str:
    start = max(0, start)
    end = min(len(seq), start + width)
    if start >= end:
        return seq
    if strategy == "mask":
        return seq[:start] + (MASK_TOKEN * (end - start)) + seq[end:]
    if strategy == "alanine":
        return seq[:start] + ("A" * (end - start)) + seq[end:]
    if strategy == "glycine":
        return seq[:start] + ("G" * (end - start)) + seq[end:]
    if strategy == "shuffle":
        block = list(seq[start:end])
        rng = np.random.default_rng(0)
        rng.shuffle(block)
        return seq[:start] + "".join(block) + seq[end:]
    return seq


def occlusion_importance(
    seq: str,
    predict_fn: Callable[[str], float],
    base_pred: Optional[float] = None,
    window: int = 5,
    strategy: str = "mask",
) -> np.ndarray:
    """
    Windowed occlusion importance distributed to positions.

    Window score at start i:
        I_i = f(x) - f(x_{[i:i+w]<-mask})

    Position-wise vector is the averaged sum of equal shares from each window
    that covers the position.
    """
    L = len(seq)
    if L == 0:
        return np.array([], dtype=float)
    if base_pred is None:
        base_pred = predict_fn(seq)

    per_pos = np.zeros(L, dtype=float)
    counts = np.zeros(L, dtype=float)

    for i in range(L):
        j = min(L, i + window)
        masked = mask_window(seq, i, j - i, strategy=strategy)
        masked_pred = predict_fn(masked)
        I_win = base_pred - masked_pred
        span = j - i
        per_pos[i:j] += I_win / span
        counts[i:j] += 1.0

    counts[counts == 0] = 1.0
    return per_pos / counts


def cip(per_pos: np.ndarray) -> np.ndarray:
    return np.cumsum(per_pos)


def cip_norm(per_pos: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(per_pos)) + 1e-12
    return np.cumsum(per_pos) / denom


# -------------------------------
# 4) Main
# -------------------------------
def clean_sequence(s: str) -> str:
    return (
        str(s).upper()
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
    )


def plot_curves(agg: pd.DataFrame, out_dir: str) -> None:
    # Ensure numeric
    for col in ["position", "mean_imp", "std_imp", "mean_cip", "std_cip"]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0.0)

    # Plot 1: Mean per-position importance
    plt.figure()
    plt.title("Average per-position occlusion importance")
    plt.xlabel("Position (aligned index)")
    plt.ylabel("Mean importance (raw)")
    x = agg["position"].values
    y = agg["mean_imp"].values
    s = agg["std_imp"].values
    plt.plot(x, y)
    plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.tight_layout()
    fig1 = os.path.join(out_dir, "/protbertLosses/Interpretation/mean_importance.png")
    plt.savefig(fig1, dpi=200)
    plt.close()

    # Plot 2: Mean CIP
    plt.figure()
    plt.title("Average Continuance Importance Profile (CIP)")
    plt.xlabel("Position (aligned index)")
    plt.ylabel("Mean CIP (raw)")
    xc = agg["position"].values
    yc = agg["mean_cip"].values
    sc = agg["std_cip"].values
    plt.plot(xc, yc)
    plt.fill_between(xc, yc - sc, yc + sc, alpha=0.2)
    plt.tight_layout()
    fig2 = os.path.join(out_dir, "/protbertLosses/Interpretation/mean_cip.png")
    plt.savefig(fig2, dpi=200)
    plt.close()

    print("Saved figures:\n ", fig1, "\n ", fig2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points_csv", required=True, help="CSV containing at least a 'Sequence' column.")
    ap.add_argument("--target_col", default=None, help="Optional ground-truth column (only saved for reference).")
    ap.add_argument("--window", type=int, default=5, help="Sliding window width.")
    ap.add_argument("--strategy", type=str, default="mask", choices=["mask", "alanine", "glycine", "shuffle"])
    ap.add_argument("--out_dir", default="cip_outputs", help="Output directory.")
    ap.add_argument("--use_surrogate", action="store_true", help="Use Ridge composition surrogate instead of grammar_predict().")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffles (if used).")
    ap.add_argument("--no_plots", action="store_true", help="Skip generating PNG plots.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load & clean
    points = pd.read_csv(args.points_csv)
    if "Sequence" not in points.columns:
        raise ValueError("Input CSV must contain a 'Sequence' column.")
    df = points.copy()
    df["Sequence"] = df["Sequence"].map(clean_sequence)
    # Map non-standard tokens to X
    df["Sequence"] = df["Sequence"].str.replace(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", regex=True)

    # Attach reference target col if provided
    if args.target_col and args.target_col in df.columns:
        df["__target__"] = pd.to_numeric(df[args.target_col], errors="coerce")
    else:
        df["__target__"] = np.nan

    # Build predictor
    if args.use_surrogate:
        # choose target col for surrogate if needed
        if df["__target__"].isna().any():
            num_cols = [c for c in df.columns if c != "Sequence" and pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError("--use_surrogate needs a numeric target column; supply --target_col.")
            df["__target__"] = pd.to_numeric(df[num_cols[0]], errors="coerce")
        predict_fn = build_surrogate_predictor(df, seed=args.seed)
    else:
        predict_fn = grammar_predict  # user must implement

    # Run per sequence
    max_len = int(df["Sequence"].str.len().max())
    rec_rows: List[Dict] = []
    cip_rows: List[Dict] = []

    def pad(arr: np.ndarray, L: int) -> np.ndarray:
        out = np.zeros(L, dtype=float)
        out[: min(L, arr.size)] = arr[: min(L, arr.size)]
        return out

    for idx, row in df.iterrows():
        seq = row["Sequence"]
        base = float(predict_fn(seq))
        imp = occlusion_importance(seq, predict_fn, base_pred=base, window=args.window, strategy=args.strategy)
        impn = imp / (np.sum(np.abs(imp)) + 1e-12)
        c = cip(imp)
        cn = cip_norm(imp)

        for pos in range(len(seq)):
            rec_rows.append(
                {
                    "sequence_index": int(idx),
                    "position": int(pos + 1),
                    "importance_raw": float(imp[pos]),
                    "importance_norm": float(impn[pos]),
                    "cip_raw": float(c[pos]),
                    "cip_norm": float(cn[pos]),
                    "seq_len": int(len(seq)),
                    "base_pred": float(base),
                    "target": float(row["__target__"]) if pd.notna(row["__target__"]) else np.nan,
                }
            )

        L = len(seq)
        imp_pad = pad(imp, max_len)
        impn_pad = pad(impn, max_len)
        cip_pad = pad(c, max_len)
        cipn_pad = pad(cn, max_len)
        for pos in range(max_len):
            cip_rows.append(
                {
                    "sequence_index": int(idx),
                    "position": int(pos + 1),
                    "imp_padded": float(imp_pad[pos]),
                    "imp_norm_padded": float(impn_pad[pos]),
                    "cip_raw_padded": float(cip_pad[pos]),
                    "cip_norm_padded": float(cipn_pad[pos]),
                    "valid": bool(pos < L),
                }
            )

    # Save detailed per-position CSV
    imp_df = pd.DataFrame(rec_rows)
    cip_long_df = pd.DataFrame(cip_rows)

    imp_path = os.path.join(args.out_dir, "/protbertLosses/Interpretation/occlusion_importance.csv")
    cip_path = os.path.join(args.out_dir, "/protbertLosses/Interpretation/continuance_profiles.csv")
    imp_df.to_csv(imp_path, index=False)
    cip_long_df.to_csv(cip_path, index=False)

    # Aggregate mean±std by aligned position (only over valid slots)
    agg = (
        cip_long_df[cip_long_df["valid"]]
        .groupby("position")
        .agg(
            mean_imp=("imp_padded", "mean"),
            std_imp=("imp_padded", "std"),
            mean_imp_norm=("imp_norm_padded", "mean"),
            std_imp_norm=("imp_norm_padded", "std"),
            mean_cip=("cip_raw_padded", "mean"),
            std_cip=("cip_raw_padded", "std"),
            mean_cip_norm=("cip_norm_padded", "mean"),
            std_cip_norm=("cip_norm_padded", "std"),
            n=("sequence_index", "count"),
        )
        .reset_index()
    )
    agg_path = os.path.join(args.out_dir, "/protbertLosses/Interpretation/summary_importance.csv")
    agg.to_csv(agg_path, index=False)

    print("Wrote:")
    print(" ", imp_path)
    print(" ", cip_path)
    print(" ", agg_path)

    if not args.no_plots:
        plot_curves(agg, args.out_dir)


if __name__ == "__main__":
    main()
