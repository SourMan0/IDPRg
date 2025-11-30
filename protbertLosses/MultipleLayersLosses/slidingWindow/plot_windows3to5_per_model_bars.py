#!/usr/bin/env python3
"""
plot_windows3to5_per_model_bars.py

For each model_X_window{3,4,5}_aa_enrichment.csv in a results directory,
make per-model bar plots for a chosen metric.

Default metric: delta_log2_enrich = log2_enrich_top - log2_enrich_bottom.
"""

import argparse
from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing model_*_windowX_aa_enrichment.csv files",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for per-model bar plots",
    )
    ap.add_argument(
        "--value_col",
        default="delta_log2_enrich",
        help=(
            "Column to plot. Default: 'delta_log2_enrich'. "
            "You can also choose 'mean_effect', 'log2_enrich_top', etc."
        ),
    )
    return ap.parse_args()


def ensure_delta_log2_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """If needed, compute delta_log2_enrich = log2_enrich_top - log2_enrich_bottom."""
    if "delta_log2_enrich" not in df.columns:
        if "log2_enrich_top" in df.columns and "log2_enrich_bottom" in df.columns:
            df = df.copy()
            df["delta_log2_enrich"] = df["log2_enrich_top"] - df["log2_enrich_bottom"]
            print("  Computed 'delta_log2_enrich'.")
        else:
            print("  Warning: cannot compute delta_log2_enrich.")
    return df


def make_barplot_for_model(csv_path: Path, out_path: Path, value_col: str):
    df = pd.read_csv(csv_path)
    print(f"Reading {csv_path.name}, columns: {list(df.columns)}")

    df = ensure_delta_log2_enrich(df)

    if "AA" not in df.columns:
        raise KeyError(f"Column 'AA' missing in {csv_path.name}")
    if value_col not in df.columns:
        raise KeyError(
            f"Column '{value_col}' missing in {csv_path.name}. Available: {list(df.columns)}"
        )

    df_sorted = df.sort_values(value_col)
    aas = df_sorted["AA"].values
    vals = df_sorted[value_col].values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(aas, vals)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel(value_col)
    ax.set_title(f"{csv_path.stem}: {value_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loop over window sizes 3, 4, 5
    for w in [3, 4, 5]:
        pattern = str(results_dir / f"model_*_window{w}_aa_enrichment.csv")
        files = sorted(glob.glob(pattern))
        print(f"\n=== Window {w}: Found {len(files)} enrichment files ===")

        if not files:
            print(f"No files matched pattern: {pattern}")
            continue

        window_out_dir = out_dir / f"window{w}"
        window_out_dir.mkdir(exist_ok=True)

        for f in files:
            fpath = Path(f)
            out_png = window_out_dir / f"{fpath.stem}_{args.value_col}.png"
            make_barplot_for_model(fpath, out_png, args.value_col)


if __name__ == "__main__":
    main()