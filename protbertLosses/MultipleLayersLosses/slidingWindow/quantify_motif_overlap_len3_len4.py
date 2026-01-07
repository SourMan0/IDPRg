#!/usr/bin/env python3
"""
quantify_motif_overlap_len3_len4.py

Quantify overlap between length-3 and length-4 motifs and make some plots.

Overlap definition:
    A 3-mer overlaps a 4-mer if the 3-mer appears as a contiguous substring
    of the 4-mer (positions 1–3 or 2–4).

Inputs
------
- len3_summary: CSV with at least column 'motif' (3-mers)
- len4_summary: CSV with at least column 'motif' (4-mers)

Typical columns from your scripts:
    motif, mean_effect, mean_abs_effect, count, score

Outputs
-------
CSV:
  - <out_prefix>_pairs.csv
      motif3, motif4,
      mean_effect_3, mean_abs_effect_3, count_3, score_3,
      mean_effect_4, mean_abs_effect_4, count_4, score_4

  - <out_prefix>_per3_summary.csv
      motif3, n_len4, total_count_4, mean_mean_effect_4, mean_score_4,
      (optionally mean_effect_3, score_3 if present)

Plots (PNG):
  - <out_prefix>_per3_nlen4_hist.png
      Histogram of how many 4-mers each 3-mer overlaps (n_len4)

  - <out_prefix>_effect_scatter.png
      Scatter of mean_effect_3 vs mean_mean_effect_4 for overlapping 3-mers
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--len3_summary",
        required=True,
        help="Path to motif_len3_all_models_summary.csv",
    )
    ap.add_argument(
        "--len4_summary",
        required=True,
        help="Path to motif_len4_all_models_summary.csv",
    )
    ap.add_argument(
        "--out_prefix",
        default="motif_len3_len4_overlap",
        help="Prefix for output CSVs/plots (default: motif_len3_len4_overlap)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    len3_df = pd.read_csv(args.len3_summary)
    len4_df = pd.read_csv(args.len4_summary)

    # Make sure the required column exists
    if "motif" not in len3_df.columns or "motif" not in len4_df.columns:
        raise ValueError("Both input CSVs must contain a 'motif' column.")

    # Deduplicate by motif, just in case
    len3_df = len3_df.drop_duplicates(subset=["motif"]).reset_index(drop=True)
    len4_df = len4_df.drop_duplicates(subset=["motif"]).reset_index(drop=True)

    len3_df_indexed = len3_df.set_index("motif")
    len4_df_indexed = len4_df.set_index("motif")

    len3_motifs = set(len3_df_indexed.index)
    len4_motifs = set(len4_df_indexed.index)

    overlap_records = []

    # For each 4-mer, check its two internal 3-mer substrings
    for motif4 in len4_motifs:
        if len(motif4) != 4:
            # Safety check if anything weird slipped into the CSV
            continue

        sub1 = motif4[0:3]
        sub2 = motif4[1:4]

        for motif3 in (sub1, sub2):
            if motif3 in len3_motifs:
                row3 = len3_df_indexed.loc[motif3]
                row4 = len4_df_indexed.loc[motif4]

                def get_val(row, col):
                    return row[col] if col in row.index else None

                overlap_records.append(
                    {
                        "motif3": motif3,
                        "motif4": motif4,
                        "mean_effect_3": get_val(row3, "mean_effect"),
                        "mean_abs_effect_3": get_val(row3, "mean_abs_effect"),
                        "count_3": get_val(row3, "count"),
                        "score_3": get_val(row3, "score"),
                        "mean_effect_4": get_val(row4, "mean_effect"),
                        "mean_abs_effect_4": get_val(row4, "mean_abs_effect"),
                        "count_4": get_val(row4, "count"),
                        "score_4": get_val(row4, "score"),
                    }
                )

    if not overlap_records:
        print("No overlaps found between 3-mers and 4-mers (by substring).")
        return

    overlap_df = pd.DataFrame(overlap_records)

    # High-level stats
    unique_len3 = len(len3_motifs)
    unique_len4 = len(len4_motifs)
    len3_with_overlap = overlap_df["motif3"].nunique()
    len4_with_overlap = overlap_df["motif4"].nunique()

    print("=== Motif overlap summary (3-mers vs 4-mers) ===")
    print(f"# unique 3-mer motifs: {unique_len3}")
    print(f"# unique 4-mer motifs: {unique_len4}")
    print(f"# 3-mer motifs that appear in at least one 4-mer: {len3_with_overlap}")
    print(f"# 4-mer motifs that contain at least one 3-mer: {len4_with_overlap}")
    if unique_len3 > 0:
        print(f"Fraction of 3-mers with overlap: {len3_with_overlap / unique_len3:.3f}")
    if unique_len4 > 0:
        print(f"Fraction of 4-mers with overlap: {len4_with_overlap / unique_len4:.3f}")

    # Per-3-mer overlap summary
    per3 = (
        overlap_df.groupby("motif3")
        .agg(
            n_len4=("motif4", "nunique"),
            total_count_4=("count_4", "sum"),
            mean_mean_effect_4=("mean_effect_4", "mean"),
            mean_score_4=("score_4", "mean"),
        )
        .reset_index()
    )

    # Optionally add mean_effect_3 and score_3 from len3 summary if present
    if "mean_effect" in len3_df.columns:
        per3 = per3.merge(
            len3_df[["motif", "mean_effect"]].rename(
                columns={"motif": "motif3", "mean_effect": "mean_effect_3"}
            ),
            on="motif3",
            how="left",
        )
    if "score" in len3_df.columns:
        per3 = per3.merge(
            len3_df[["motif", "score"]].rename(
                columns={"motif": "motif3", "score": "score_3"}
            ),
            on="motif3",
            how="left",
        )

    # ---- Save CSVs ----
    out_base = Path(args.out_prefix)
    pairs_path = f"{out_base}_pairs.csv"
    per3_path = f"{out_base}_per3_summary.csv"

    overlap_df.to_csv(pairs_path, index=False)
    per3.to_csv(per3_path, index=False)

    print()
    print(f"Saved overlapping motif pairs to: {pairs_path}")
    print(f"Saved per-3-mer overlap summary to: {per3_path}")

    # ---- Plots ----

    # 1) Histogram: how many 4-mers each 3-mer overlaps
    plt.figure()
    per3["n_len4"].hist(bins=20)
    plt.xlabel("Number of overlapping 4-mers per 3-mer (n_len4)")
    plt.ylabel("Count of 3-mers")
    plt.title("Distribution of 4-mer overlaps per 3-mer")
    hist_path = f"{out_base}_per3_nlen4_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved histogram plot to: {hist_path}")

    # 2) Scatter: mean_effect_3 vs mean_mean_effect_4 (if available)
    if "mean_effect_3" in per3.columns and "mean_mean_effect_4" in per3.columns:
        plt.figure()
        plt.scatter(per3["mean_effect_3"], per3["mean_mean_effect_4"], alpha=0.7)
        plt.xlabel("mean_effect_3 (3-mer)")
        plt.ylabel("mean_mean_effect_4 (overlapping 4-mers)")
        plt.title("3-mer vs overlapping 4-mer mean effects")
        scatter_path = f"{out_base}_effect_scatter.png"
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        print(f"Saved effect scatter plot to: {scatter_path}")
    else:
        print("Skipping effect scatter plot (mean_effect columns not found).")


if __name__ == "__main__":
    main()