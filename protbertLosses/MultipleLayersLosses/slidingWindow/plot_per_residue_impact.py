#!/usr/bin/env python3
"""
plot_per_residue_impact.py

Make a "per-residue impact on predicted Rg" plot for a single sequence,
using sliding-window occlusion results saved by apply_sliding_window_to_all_protbert.py.

Assumes files in a model results directory:
  - effects_by_window.pkl
  - windows_by_window.pkl

Each window size is turned into per-residue contributions and plotted as
a grouped bar chart over residue index.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        required=True,
        help="Directory for a single model, e.g. protbert_sliding_results_top5/model_5",
    )
    ap.add_argument(
        "--seq_index",
        type=int,
        default=0,
        help="Which sequence index to plot (0-based; same order as seq_csv).",
    )
    ap.add_argument(
        "--windows",
        default="3,4,5",
        help="Comma-separated list of window sizes to include, e.g. '3,4,5'.",
    )
    ap.add_argument(
        "--out_png",
        default="per_residue_impact.png",
        help="Output PNG filename.",
    )
    return ap.parse_args()


def per_residue_from_windows(effects, windows, L):
    """
    Convert per-window effects into per-residue contributions.

    effects : array-like, shape (num_windows,)
    windows : list of (start, end) tuples
    L       : sequence length

    Strategy: distribute each window's effect evenly across residues in that window,
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

    # avoid division by zero
    mask = counts > 0
    contrib[mask] /= counts[mask]
    return contrib


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    with open(results_dir / "effects_by_window.pkl", "rb") as f:
        effects_by_window = pickle.load(f)

    with open(results_dir / "windows_by_window.pkl", "rb") as f:
        windows_by_window = pickle.load(f)

    # Determine the actual window size for each index (0,1,2,...) by looking
    # at the span of the first window in the first sequence.
    window_lengths = []
    for idx, win_lists in enumerate(windows_by_window):
        seq0_windows = win_lists[0]
        if not seq0_windows:
            raise ValueError(f"No windows stored for index {idx}.")
        start0, end0 = seq0_windows[0]
        window_lengths.append(end0 - start0)

    length_to_idx = {L: i for i, L in enumerate(window_lengths)}

    requested_windows = [int(w.strip()) for w in args.windows.split(",")]
    print("Available window sizes:", window_lengths)
    print("Requested window sizes:", requested_windows)

    # Get sequence length from any window list
    # (last window's end should equal sequence length)
    any_win_list = windows_by_window[0][args.seq_index]
    seq_len = any_win_list[-1][1]
    x = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(12, 4))

    for w in requested_windows:
        if w not in length_to_idx:
            print(f"Window {w} not found in results; skipping.")
            continue

        w_idx = length_to_idx[w]
        eff_list = effects_by_window[w_idx][args.seq_index]
        win_list = windows_by_window[w_idx][args.seq_index]

        eff_arr = np.asarray(eff_list, dtype=float)
        contrib = per_residue_from_windows(eff_arr, win_list, seq_len)

        # bar plot for this window size; matplotlib will auto-assign colors
        ax.bar(x, contrib, alpha=0.6, label=f"window={w}")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("ΔRg (prediction change)")
    ax.set_title("Per-residue impact on predicted Rg")
    ax.legend()
    fig.tight_layout()

    fig.savefig(args.out_png, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()
