#!/usr/bin/env python3
"""
plot_per_residue_impact_mean_models.py

Compute and plot the *average* per-residue impact on predicted Rg,
averaged across:

  - multiple models: model_*/ directories under --results_root
  - multiple window sizes (default: 3,4,5)

Assumes each model directory contains:
  - effects_by_window.pkl
  - windows_by_window.pkl

These are produced by apply_sliding_window_to_all_protbert.py.
"""

import argparse
from pathlib import Path
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        required=True,
        help="Root directory containing model_* subdirs "
             "(e.g. protbert_sliding_results_top5)",
    )
    ap.add_argument(
        "--seq_index",
        type=int,
        default=0,
        help="Which sequence index to plot (0-based, same as in seq_csv).",
    )
    ap.add_argument(
        "--windows",
        default="3,4,5",
        help="Comma-separated window sizes to include, e.g. '3,4,5'.",
    )
    ap.add_argument(
        "--out_png",
        default="per_residue_impact_mean_models.png",
        help="Output PNG filename.",
    )
    ap.add_argument(
        "--out_csv",
        default="per_residue_impact_mean_models.csv",
        help="Optional CSV with mean per-residue impacts.",
    )
    return ap.parse_args()


def per_residue_from_windows(effects, windows, L):
    """
    Convert per-window effects into per-residue contributions.

    effects : array-like of shape (num_windows,)
    windows : list of (start, end) tuples
    L       : sequence length

    Strategy: distribute each window's effect evenly over its residues,
    then average over the number of windows covering each residue.
    """
    contrib = np.zeros(L, dtype=float)
    counts = np.zeros(L, dtype=float)

    for eff, (start, end) in zip(effects, windows):
        span = end - start
        if span <= 0:
            continue
        per_res = eff / span
        contrib[start:end] += per_res
        counts[start:end] += 1.0

    mask = counts > 0
    contrib[mask] /= counts[mask]
    return contrib


def load_window_lengths(windows_by_window):
    """Infer window length for each index in windows_by_window."""
    lens = []
    for win_lists in windows_by_window:
        seq0 = win_lists[0]
        if not seq0:
            raise ValueError("Found empty window list when inferring lengths.")
        start0, end0 = seq0[0]
        lens.append(end0 - start0)
    return lens


def main():
    args = parse_args()
    root = Path(args.results_root)
    requested_windows = [int(w.strip()) for w in args.windows.split(",")]

    model_dirs = sorted(d for d in root.glob("model_*") if d.is_dir())
    if not model_dirs:
        raise RuntimeError(f"No model_* dirs found under {root}")

    print("Models found:", [d.name for d in model_dirs])
    print("Requested windows:", requested_windows)

    contribs_all = []
    seq_len = None

    for mdir in model_dirs:
        print(f"\n=== Processing {mdir.name} ===")
        with open(mdir / "effects_by_window.pkl", "rb") as f:
            effects_by_window = pickle.load(f)
        with open(mdir / "windows_by_window.pkl", "rb") as f:
            windows_by_window = pickle.load(f)

        window_lengths = load_window_lengths(windows_by_window)
        length_to_idx = {L: i for i, L in enumerate(window_lengths)}
        print("  Available window sizes:", window_lengths)

        # determine sequence length from any window list
        seq_windows = windows_by_window[0][args.seq_index]
        this_seq_len = seq_windows[-1][1]
        if seq_len is None:
            seq_len = this_seq_len
        elif seq_len != this_seq_len:
            raise ValueError(
                f"Sequence length mismatch across models "
                f"({seq_len} vs {this_seq_len})"
            )

        for w in requested_windows:
            if w not in length_to_idx:
                print(f"  Window {w} not present in {mdir.name}; skipping.")
                continue

            w_idx = length_to_idx[w]
            eff_list = effects_by_window[w_idx][args.seq_index]
            win_list = windows_by_window[w_idx][args.seq_index]

            eff_arr = np.asarray(eff_list, dtype=float)
            contrib = per_residue_from_windows(eff_arr, win_list, seq_len)
            contribs_all.append(contrib)

    if not contribs_all:
        raise RuntimeError("No contributions collected; check windows/models.")

    contribs_all = np.stack(contribs_all, axis=0)
    mean_contrib = contribs_all.mean(axis=0)

    # Save CSV
    x = np.arange(seq_len)
    import pandas as pd  # imported here to keep top clean

    df_out = pd.DataFrame({"res_index": x, "mean_delta_Rg": mean_contrib})
    df_out.to_csv(args.out_csv, index=False)
    print(f"\nSaved mean per-residue impacts to {args.out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, mean_contrib, width=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Mean ΔRg (prediction change)")
    ax.set_title(
        f"Mean per-residue impact on predicted Rg\n"
        f"(avg over models & windows {requested_windows})"
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()
