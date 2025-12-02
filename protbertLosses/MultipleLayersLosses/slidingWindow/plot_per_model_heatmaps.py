#!/usr/bin/env python3
"""
plot_per_model_heatmaps.py

For each model, build a heatmap over:
    y-axis: amino acids
    x-axis: window sizes
    value: chosen metric (default: delta_log2_enrich)

Assumes CSV files named:
    model_{i}_window{W}_aa_enrichment.csv

and columns:
    'AA',
    'log2_enrich_top',
    'log2_enrich_bottom',
    'mean_effect', etc.
"""

import argparse
from pathlib import Path
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        required=True,
        help="Directory with model_*_window*_aa_enrichment.csv files.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save heatmap PNGs.",
    )
    ap.add_argument(
        "--value_col",
        default="delta_log2_enrich",
        help=(
            "Which column to plot. Default: 'delta_log2_enrich' = "
            "log2_enrich_top - log2_enrich_bottom.\n"
            "You may also use 'mean_effect', 'log2_enrich_top', etc."
        ),
    )
    return ap.parse_args()


def ensure_delta_log2_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 'delta_log2_enrich' column.
    If missing but 'log2_enrich_top' and 'log2_enrich_bottom' exist,
    compute it as top - bottom.
    """
    if "delta_log2_enrich" not in df.columns:
        if "log2_enrich_top" in df.columns and "log2_enrich_bottom" in df.columns:
            df = df.copy()
            df["delta_log2_enrich"] = df["log2_enrich_top"] - df["log2_enrich_bottom"]
            print("  Computed 'delta_log2_enrich' = log2_enrich_top - log2_enrich_bottom.")
        else:
            print(
                "  Warning: cannot compute 'delta_log2_enrich' because "
                "'log2_enrich_top' and/or 'log2_enrich_bottom' are missing."
            )
    return df


def collect_files_by_model(results_dir: Path):
    """
    Scan results_dir and group files by model_id.
    Returns: dict[model_id] -> list of (window, Path)
    """
    pattern = str(results_dir / "model_*_window*_aa_enrichment.csv")
    files = glob.glob(pattern)
    by_model = {}

    for f in files:
        fname = Path(f).name
        m = re.match(r"model_(\d+)_window(\d+)_aa_enrichment\.csv$", fname)
        if not m:
            continue
        model_id = int(m.group(1))
        window = int(m.group(2))
        by_model.setdefault(model_id, []).append((window, Path(f)))

    return by_model


def build_matrix_for_model(window_files, value_col):
    """
    Given list[(window, Path)], load each CSV and build:
        - windows_sorted: sorted list of window sizes
        - mat: np.array shape (len(AA_ORDER), n_windows)
    """
    windows_sorted = sorted(w for w, _ in window_files)
    n_win = len(windows_sorted)
    n_aa = len(AA_ORDER)

    mat = np.full((n_aa, n_win), np.nan, dtype=float)

    for j, w in enumerate(windows_sorted):
        csv_path = next(p for ww, p in window_files if ww == w)
        df = pd.read_csv(csv_path)
        print(f"  Reading {csv_path.name} with columns: {list(df.columns)}")

        if "AA" not in df.columns:
            raise KeyError(
                f"Expected column 'AA' in {csv_path.name}, found: {list(df.columns)}"
            )

        df = ensure_delta_log2_enrich(df)

        if value_col not in df.columns:
            raise KeyError(
                f"Requested value_col '{value_col}' not found in {csv_path.name}.\n"
                f"Available columns: {list(df.columns)}"
            )

        for i, aa in enumerate(AA_ORDER):
            rows = df[df["AA"] == aa]
            if not rows.empty:
                mat[i, j] = rows[value_col].iloc[0]

    return windows_sorted, mat


def plot_heatmap(model_id, windows, mat, out_path: Path, value_col: str):
    fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(AA_ORDER)))
    ax.set_yticklabels(AA_ORDER)
    ax.set_ylabel("Amino acid")

    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_xlabel("Window size")

    ax.set_title(f"Model {model_id}: {value_col}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_model = collect_files_by_model(results_dir)
    model_ids = sorted(by_model.keys())
    print(f"Found models: {model_ids}")

    if not model_ids:
        print("No model_*_window*_aa_enrichment.csv files found.")
        return

    for model_id in model_ids:
        print(f"\n=== Model {model_id} ===")
        window_files = by_model[model_id]
        windows, mat = build_matrix_for_model(window_files, args.value_col)

        out_path = out_dir / f"model_{model_id}_heatmap_{args.value_col}.png"
        plot_heatmap(model_id, windows, mat, out_path, args.value_col)


if __name__ == "__main__":
    main()
