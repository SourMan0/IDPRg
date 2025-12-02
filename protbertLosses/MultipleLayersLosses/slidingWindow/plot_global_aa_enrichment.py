#!/usr/bin/env python3
"""
plot_global_aa_enrichment.py

Make global visualizations from global_aa_enrichment_summary.csv
produced by analyze_all_models.py.

It can work with any numeric column, but by default it uses a derived
metric:

    delta_log2_enrich = log2_enrich_top - log2_enrich_bottom

which measures enrichment in "top" windows vs "bottom" windows.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--global_csv",
        required=True,
        help="Path to global_aa_enrichment_summary.csv",
    )
    ap.add_argument(
        "--out_prefix",
        required=True,
        help="Prefix for output plots (e.g. protbert_sliding_summary_top5/global_aa_enrichment)",
    )
    ap.add_argument(
        "--value_col",
        default="delta_log2_enrich",
        help=(
            "Which column to plot. Default: 'delta_log2_enrich' "
            "(computed from log2_enrich_top - log2_enrich_bottom "
            "if not already present). You can also set this to e.g. "
            "'mean_effect', 'mean_effect_all', 'log2_enrich_top', etc."
        ),
    )
    return ap.parse_args()


def ensure_delta_log2_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 'delta_log2_enrich' column.
    If it doesn't but has 'log2_enrich_top' and 'log2_enrich_bottom',
    compute it as top - bottom.
    """
    if "delta_log2_enrich" not in df.columns:
        if "log2_enrich_top" in df.columns and "log2_enrich_bottom" in df.columns:
            df = df.copy()
            df["delta_log2_enrich"] = df["log2_enrich_top"] - df["log2_enrich_bottom"]
            print("Computed 'delta_log2_enrich' = log2_enrich_top - log2_enrich_bottom.")
        else:
            print(
                "Warning: cannot compute 'delta_log2_enrich' because "
                "'log2_enrich_top' and/or 'log2_enrich_bottom' are missing."
            )
    return df


def make_heatmap(df, value_col, out_png):
    """
    Heatmap of AA vs window, average across models.
    """
    # Aggregate across models
    grp = df.groupby(["AA", "window"])[value_col].mean().reset_index()

    # Pivot to AA (rows) x window (columns)
    heat = grp.pivot(index="AA", columns="window", values=value_col)

    # Sort windows numerically, AAs alphabetically
    heat = heat.sort_index(axis=0)
    heat = heat.reindex(sorted(heat.columns), axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heat.values, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_xlabel("Window size")

    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_ylabel("Amino acid")

    ax.set_title(f"Global AA {value_col} (mean across models)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[saved] Heatmap → {out_png}")


def make_aa_barplot(df, value_col, out_png):
    """
    Bar plot: per-AA mean value across windows & models.
    """
    aa_mean = df.groupby("AA")[value_col].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(aa_mean.index, aa_mean.values)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel(value_col)
    ax.set_title(f"Mean {value_col} per AA (global)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[saved] AA bar plot → {out_png}")


def make_window_lineplot(df, value_col, out_png):
    """
    Line plot: average value vs window size (across AA & models).
    """
    win_mean = df.groupby("window")[value_col].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(win_mean.index, win_mean.values, marker="o")
    ax.set_xlabel("Window size")
    ax.set_ylabel(value_col)
    ax.set_title(f"Mean {value_col} vs window size")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[saved] Window line plot → {out_png}")


def main():
    args = parse_args()

    df = pd.read_csv(args.global_csv)
    print(f"Loaded {len(df)} rows from {args.global_csv}")
    print("Columns:", list(df.columns))

    # Ensure we have delta_log2_enrich if needed
    df = ensure_delta_log2_enrich(df)

    # Check that the requested value_col exists
    if args.value_col not in df.columns:
        raise KeyError(
            f"Requested value_col '{args.value_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    out_prefix_path = Path(args.out_prefix)
    out_dir = out_prefix_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Heatmap AA x window
    make_heatmap(
        df,
        args.value_col,
        out_dir / f"{out_prefix_path.name}_heatmap_{args.value_col}.png",
    )

    # 2) Bar plot per AA
    make_aa_barplot(
        df,
        args.value_col,
        out_dir / f"{out_prefix_path.name}_AA_bar_{args.value_col}.png",
    )

    # 3) Line plot vs window
    make_window_lineplot(
        df,
        args.value_col,
        out_dir / f"{out_prefix_path.name}_window_line_{args.value_col}.png",
    )


if __name__ == "__main__":
    main()
