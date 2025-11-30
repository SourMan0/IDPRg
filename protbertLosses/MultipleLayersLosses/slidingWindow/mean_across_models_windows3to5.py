#!/usr/bin/env python3
"""
mean_across_models_windows3to5.py

Compute the mean delta_log2_enrich across:
    - all models (model_1 ... model_5)
    - window sizes 3, 4, 5

Assumes files named:
    model_X_windowY_aa_enrichment.csv
in a results directory (e.g., protbert_sliding_summary_top5).
"""

import argparse
from pathlib import Path
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing model_*_windowX_aa_enrichment.csv files",
    )
    ap.add_argument(
        "--out_csv",
        default="mean_models_windows3to5.csv",
        help="Output CSV file name",
    )
    ap.add_argument(
        "--out_png",
        default="mean_models_windows3to5.png",
        help="Output PNG plot file name",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # windows to use
    windows = [3, 4, 5]

    dfs = []

    for w in windows:
        pattern = str(results_dir / f"model_*_window{w}_aa_enrichment.csv")
        files = sorted(glob.glob(pattern))

        print(f"Window {w}: found {len(files)} files.")
        for f in files:
            df = pd.read_csv(f)
            # ensure delta_log2_enrich exists
            if "delta_log2_enrich" not in df.columns:
                if "log2_enrich_top" in df.columns and "log2_enrich_bottom" in df.columns:
                    df["delta_log2_enrich"] = df["log2_enrich_top"] - df["log2_enrich_bottom"]
                else:
                    raise KeyError(f"{f} missing needed columns for delta_log2_enrich.")

            df_small = df[["AA", "delta_log2_enrich"]].copy()
            df_small["model"] = Path(f).stem.split("_")[1]
            df_small["window"] = w
            dfs.append(df_small)

    # combine all
    all_df = pd.concat(dfs, ignore_index=True)

    # group by AA: compute mean across models × windows
    mean_df = (
        all_df.groupby("AA")["delta_log2_enrich"]
        .mean()
        .reset_index()
        .sort_values("delta_log2_enrich")
    )

    # save CSV
    mean_df.to_csv(args.out_csv, index=False)
    print(f"Saved mean CSV → {args.out_csv}")

    # make barplot
    plt.figure(figsize=(10,5))
    plt.bar(mean_df["AA"], mean_df["delta_log2_enrich"])
    plt.xlabel("Amino acid")
    plt.ylabel("Mean delta_log2_enrich")
    plt.title("Mean delta_log2_enrich across models (windows 3–5)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()

    print(f"Saved plot → {args.out_png}")


if __name__ == "__main__":
    main()