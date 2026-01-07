#!/usr/bin/env python3
"""
motif_delta_log2_enrichment.py

Compute motif-level Δlog2 enrichment for k-mers (motif_len = 3 or 4)
using ProtBERT sliding-window occlusion results.

For a given motif length k:

  - Collect all window fragments of length k (from frags_by_window.pkl)
    and their corresponding window effects (from effects_by_window.pkl),
    across ALL models under --results_root.
  - Rank all windows by effect.
  - Define top and bottom fractions (top_frac, bottom_frac).
  - For each motif m, compute:

      freq_top(m)    = count of motif m in top windows / top_n
      freq_bottom(m) = count of motif m in bottom windows / bottom_n

      delta_log2(m) = log2( (freq_top(m) + eps) / (freq_bottom(m) + eps) )

  - Write a CSV with motif-level stats.
  - Make a barplot of top motifs by delta_log2.

This is the "motif version" of your AA-enrichment bar plot.

Example:
    python3 motif_delta_log2_enrichment.py \
      --results_root protbert_sliding_results_top5 \
      --motif_len 4 \
      --top_frac 0.15 \
      --bottom_frac 0.15 \
      --min_count 10 \
      --topk_plot 30 \
      --out_csv motif_len4_delta_log2.csv \
      --out_png motif_len4_delta_log2_bar.png
"""

import argparse
from pathlib import Path
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        required=True,
        help="Root directory containing model_* subdirs with sliding-window results.",
    )
    ap.add_argument(
        "--motif_len",
        type=int,
        default=4,
        help="Motif length k (typically 3 or 4).",
    )
    ap.add_argument(
        "--top_frac",
        type=float,
        default=0.15,
        help="Fraction of windows to treat as 'top' (most positive effects).",
    )
    ap.add_argument(
        "--bottom_frac",
        type=float,
        default=0.15,
        help="Fraction of windows to treat as 'bottom' (most negative effects).",
    )
    ap.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum total count of a motif to be included in the output.",
    )
    ap.add_argument(
        "--topk_plot",
        type=int,
        default=30,
        help="Number of top motifs (by delta_log2) to show in the barplot.",
    )
    ap.add_argument(
        "--out_csv",
        default="motif_delta_log2_enrichment.csv",
        help="Output CSV file with motif-level stats.",
    )
    ap.add_argument(
        "--out_png",
        default="motif_delta_log2_enrichment_bar.png",
        help="Output PNG file for the barplot.",
    )
    return ap.parse_args()


def find_k_window_index(frags_by_window: List, motif_len: int) -> int:
    """
    Given frags_by_window (list over window sizes), find the index whose
    fragments have length == motif_len.

    We scan each window index and look in the first non-empty sequence
    for a fragment; once we see one of length motif_len, we return that index.

    Raises ValueError if no such window index is found.
    """
    for w_idx, frags_per_seq in enumerate(frags_by_window):
        # frags_per_seq is a list: one entry per sequence
        for seq_frags in frags_per_seq:
            if len(seq_frags) == 0:
                continue
            frag0 = str(seq_frags[0])
            if len(frag0) == motif_len:
                return w_idx
            else:
                # if we found a fragment but it's not the right length,
                # no need to check further sequences for this window
                break
    raise ValueError(f"No window index found with motif length {motif_len}.")


def collect_motifs_and_effects_for_model(
    model_dir: Path,
    motif_len: int,
) -> Tuple[List[str], List[float]]:
    """
    For a single model directory, load frags_by_window.pkl and
    effects_by_window.pkl, identify the window index whose fragment
    length == motif_len, and return flattened lists of motifs and effects.
    """
    effects_path = model_dir / "effects_by_window.pkl"
    frags_path = model_dir / "frags_by_window.pkl"

    if not effects_path.exists() or not frags_path.exists():
        raise FileNotFoundError(
            f"Missing effects_by_window.pkl or frags_by_window.pkl in {model_dir}"
        )

    with open(effects_path, "rb") as f:
        effects_by_window = pickle.load(f)
    with open(frags_path, "rb") as f:
        frags_by_window = pickle.load(f)

    # Find window index whose fragment length == motif_len
    w_idx = find_k_window_index(frags_by_window, motif_len)

    effects_win = effects_by_window[w_idx]
    frags_win = frags_by_window[w_idx]

    all_motifs = []
    all_effects = []

    # Each element in effects_win/frags_win is per sequence
    for eff_seq, frag_seq in zip(effects_win, frags_win):
        eff_seq = np.asarray(eff_seq, dtype=float)
        frag_seq = np.asarray(frag_seq, dtype=str)
        if len(eff_seq) != len(frag_seq):
            # Use min length to be safe
            L = min(len(eff_seq), len(frag_seq))
            eff_seq = eff_seq[:L]
            frag_seq = frag_seq[:L]
        all_motifs.extend(frag_seq.tolist())
        all_effects.extend(eff_seq.tolist())

    return all_motifs, all_effects


def main():
    args = parse_args()

    results_root = Path(args.results_root)
    model_dirs = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("model_")],
        key=lambda p: p.name,
    )

    if not model_dirs:
        raise RuntimeError(f"No model_* directories found under {results_root}")

    print("Found models:", [d.name for d in model_dirs])

    all_motifs = []
    all_effects = []

    # Collect motifs/effects from all models
    for mdir in model_dirs:
        print(f"Collecting motifs for {mdir.name} ...")
        try:
            motifs_m, effects_m = collect_motifs_and_effects_for_model(
                mdir, args.motif_len
            )
        except Exception as e:
            print(f"  [warn] Skipping {mdir.name} due to error: {e}")
            continue

        all_motifs.extend(motifs_m)
        all_effects.extend(effects_m)

    if not all_motifs:
        raise RuntimeError("No motifs collected; check motif_len or results_root.")

    # Build a DataFrame
    df = pd.DataFrame({"motif": all_motifs, "effect": all_effects})
    print(f"Total windows collected: {len(df)}")

    # Sort by effect (descending) to get top / bottom windows
    df_sorted = df.sort_values("effect", ascending=False).reset_index(drop=True)

    top_n = max(1, int(len(df_sorted) * args.top_frac))
    bottom_n = max(1, int(len(df_sorted) * args.bottom_frac))

    df_top = df_sorted.iloc[:top_n]
    df_bottom = df_sorted.iloc[-bottom_n:]

    print(f"Top windows: {top_n}, Bottom windows: {bottom_n}")

    # Global counts
    global_counts = df["motif"].value_counts()
    global_total = len(df)
    global_freq = global_counts / global_total

    # Top/bottom counts
    top_counts = df_top["motif"].value_counts()
    bottom_counts = df_bottom["motif"].value_counts()

    # Build a common index of motifs to consider
    all_motifs_set = set(global_counts.index)
    motifs_sorted = sorted(all_motifs_set)

    eps = 1e-9

    rows = []
    for m in motifs_sorted:
        count_all = int(global_counts.get(m, 0))
        if count_all < args.min_count:
            continue

        count_top = int(top_counts.get(m, 0))
        count_bottom = int(bottom_counts.get(m, 0))

        freq_all = count_all / float(global_total)
        freq_top = count_top / float(top_n) if top_n > 0 else 0.0
        freq_bottom = count_bottom / float(bottom_n) if bottom_n > 0 else 0.0

        # Δlog2 = log2(freq_top / freq_bottom)
        delta_log2 = np.log2((freq_top + eps) / (freq_bottom + eps))

        # Mean signed effect for this motif (for additional info)
        mean_effect = df.loc[df["motif"] == m, "effect"].mean()

        rows.append(
            {
                "motif": m,
                "count_all": count_all,
                "count_top": count_top,
                "count_bottom": count_bottom,
                "freq_all": freq_all,
                "freq_top": freq_top,
                "freq_bottom": freq_bottom,
                "delta_log2": delta_log2,
                "mean_effect": mean_effect,
            }
        )

    if not rows:
        raise RuntimeError("No motifs passed the min_count filter; try lowering --min_count.")

    df_out = pd.DataFrame(rows).sort_values("delta_log2", ascending=False)
    df_out.to_csv(args.out_csv, index=False)
    print(f"[saved] Motif-level enrichment CSV → {args.out_csv}")

    # --- Barplot of top motifs by delta_log2 ---
    topk = df_out.head(args.topk_plot)
    plt.figure(figsize=(max(8, 0.4 * len(topk)), 5))
    plt.bar(topk["motif"], topk["delta_log2"])
    plt.xticks(rotation=90)
    plt.xlabel("Motif")
    plt.ylabel("Δlog2 (freq_top / freq_bottom)")
    plt.title(f"Top {len(topk)} motifs by Δlog2 (motif_len={args.motif_len})")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()
    print(f"[saved] Motif barplot → {args.out_png}")


if __name__ == "__main__":
    main()
