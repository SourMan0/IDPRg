#!/usr/bin/env python3
"""
find_top_motifs_len4_all_models.py

For ALL models under --results_root, find 4-residue motifs (window length = 4)
that have the largest impact on predicted Rg (ΔRg), and aggregate across:

  - all model_* directories
  - all sequences in the seq_csv (up to the number present in the PKLs)
  - window size = 4

Outputs:
  1) A per-instance CSV:
       model, seq_index, seq_id (optional), start, end, motif, effect
  2) A per-motif summary CSV:
       motif, mean_effect, mean_abs_effect, count, score
  3) A bar plot PNG of the top motifs by |mean_effect|
"""

import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
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
        "--seq_csv",
        required=True,
        help="CSV with sequences used for sliding-window analysis.",
    )
    ap.add_argument(
        "--sequence_col",
        default="Protein Sequence",
        help="Column name in seq_csv that contains the protein sequence.",
    )
    ap.add_argument(
        "--id_col",
        default=None,
        help="Optional column name for sequence ID (e.g. 'ID', 'Name'). "
             "If not provided, seq_index will be used.",
    )
    ap.add_argument(
        "--motif_len",
        type=int,
        default=4,
        help="Motif length (window length) to analyze. Default: 4.",
    )
    ap.add_argument(
        "--topk_instances",
        type=int,
        default=500,
        help="Number of top motif *instances* to write to the instances CSV.",
    )
    ap.add_argument(
        "--topk_motifs",
        type=int,
        default=30,
        help="Number of top motifs to show in the bar plot.",
    )
    ap.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum number of occurrences for a motif to be included in summary.",
    )
    ap.add_argument(
        "--sort_mode",
        choices=["signed", "abs"],
        default="abs",
        help=(
            "How to rank motifs:\n"
            "  'signed' = sort by mean_effect (most pos/neg first)\n"
            "  'abs'    = sort by |mean_effect| (largest magnitude)."
        ),
    )
    ap.add_argument(
        "--out_instances_csv",
        default="motif_len4_all_models_instances.csv",
        help="Output CSV for top motif instances.",
    )
    ap.add_argument(
        "--out_motifs_csv",
        default="motif_len4_all_models_summary.csv",
        help="Output CSV for aggregated motif statistics.",
    )
    ap.add_argument(
        "--out_png",
        default="motif_len4_all_models_top_motifs.png",
        help="Output PNG for bar plot of top motifs.",
    )
    return ap.parse_args()


def load_window_lengths(windows_by_window):
    """Infer the window length (end-start) for each index in windows_by_window."""
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
    motif_len = args.motif_len

    # --- find all model_* dirs ---
    model_dirs = sorted(d for d in root.glob("model_*") if d.is_dir())
    if not model_dirs:
        raise RuntimeError(f"No model_* dirs found under {root}")
    print("Models found:", [d.name for d in model_dirs])

    # --- load sequences ---
    df_seq = pd.read_csv(args.seq_csv)
    if args.sequence_col not in df_seq.columns:
        raise KeyError(
            f"sequence_col='{args.sequence_col}' not found in CSV columns: "
            f"{list(df_seq.columns)}"
        )
    if args.id_col is not None and args.id_col not in df_seq.columns:
        raise KeyError(
            f"id_col='{args.id_col}' not found in CSV columns: "
            f"{list(df_seq.columns)}"
        )

    sequences = df_seq[args.sequence_col].astype(str).tolist()
    num_seqs_csv = len(sequences)
    print(f"Loaded {num_seqs_csv} sequences from {args.seq_csv}")

    all_instances = []

    # --- iterate over models ---
    for mdir in model_dirs:
        print(f"\n=== Processing {mdir.name} ===")

        with open(mdir / "effects_by_window.pkl", "rb") as f:
            effects_by_window = pickle.load(f)
        with open(mdir / "windows_by_window.pkl", "rb") as f:
            windows_by_window = pickle.load(f)

        window_lengths = load_window_lengths(windows_by_window)
        length_to_idx = {L: i for i, L in enumerate(window_lengths)}
        print("  Available window sizes:", window_lengths)

        if motif_len not in length_to_idx:
            print(f"  Motif length {motif_len} not present in {mdir.name}; skipping.")
            continue

        w_idx = length_to_idx[motif_len]
        num_seqs_windows = len(windows_by_window[w_idx])
        num_seqs_use = min(num_seqs_csv, num_seqs_windows)
        print(f"  Using {num_seqs_use} sequences (min of CSV/windows).")

        for s in range(num_seqs_use):
            seq = sequences[s]
            seq_len = len(seq)

            win_list = windows_by_window[w_idx][s]
            eff_list = effects_by_window[w_idx][s]

            if len(win_list) != len(eff_list):
                raise ValueError(
                    f"Mismatch in number of windows/effects for {mdir.name}, "
                    f"sequence {s}: {len(win_list)} vs {len(eff_list)}"
                )

            if args.id_col is not None:
                seq_id = df_seq.loc[s, args.id_col]
            else:
                seq_id = s

            for (start, end), eff in zip(win_list, eff_list):
                if end > seq_len:
                    continue
                motif = seq[start:end]
                if len(motif) != motif_len:
                    continue

                all_instances.append(
                    {
                        "model": mdir.name,
                        "seq_index": s,
                        "seq_id": seq_id,
                        "start": start,
                        "end": end,
                        "motif": motif,
                        "effect": float(eff),
                    }
                )

    if not all_instances:
        raise RuntimeError("No motif instances collected. Check inputs.")

    df_inst = pd.DataFrame(all_instances)
    print(f"\nCollected {len(df_inst)} motif instances (length={motif_len}) across all models.")

    # --- score per instance for ranking ---
    if args.sort_mode == "abs":
        df_inst["score"] = df_inst["effect"].abs()
    else:
        df_inst["score"] = df_inst["effect"]

    # --- save top instances ---
    df_top_inst = df_inst.sort_values("score", ascending=False).head(args.topk_instances)
    df_top_inst.to_csv(args.out_instances_csv, index=False)
    print(f"[saved] Top {args.topk_instances} motif instances → {args.out_instances_csv}")

    # --- aggregate per motif ---
    df_motifs = (
        df_inst.groupby("motif")
        .agg(
            mean_effect=("effect", "mean"),
            mean_abs_effect=("effect", lambda x: np.mean(np.abs(x))),
            count=("effect", "size"),
        )
        .reset_index()
    )

    # filter by count
    df_motifs = df_motifs[df_motifs["count"] >= args.min_count].copy()
    if df_motifs.empty:
        raise RuntimeError(
            f"No motifs remained after filtering with min_count={args.min_count}."
        )

    # rank motifs
    if args.sort_mode == "abs":
        df_motifs["score"] = df_motifs["mean_abs_effect"]
    else:
        df_motifs["score"] = df_motifs["mean_effect"]

    df_motifs = df_motifs.sort_values("score", ascending=False)
    df_motifs.to_csv(args.out_motifs_csv, index=False)
    print(f"[saved] Motif summary → {args.out_motifs_csv}")

    # --- bar plot of top motifs ---
    df_plot = df_motifs.head(args.topk_motifs).copy()
    x = np.arange(len(df_plot))

    plt.figure(figsize=(max(8, len(df_plot) * 0.4), 5))
    plt.bar(x, df_plot["mean_effect"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, df_plot["motif"], rotation=90)
    plt.xlabel("Motif (length = %d)" % motif_len)
    plt.ylabel("Mean ΔRg (prediction change)")
    plt.title(
        f"Top {args.topk_motifs} motifs by {args.sort_mode} mean effect\n"
        f"(aggregated over all models)"
    )
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()
    print(f"[saved] Motif bar plot → {args.out_png}")

    print("\nTop 10 motifs:")
    print(df_plot[["motif", "mean_effect", "mean_abs_effect", "count"]].head(10))


if __name__ == "__main__":
    main()
