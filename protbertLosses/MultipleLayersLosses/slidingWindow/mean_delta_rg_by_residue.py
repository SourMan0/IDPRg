#!/usr/bin/env python3
"""
mean_delta_rg_by_residue.py

Compute the mean ΔRg effect per amino acid, pooled across:

  - all model_* directories under --results_root
  - selected window sizes (e.g. windows 3–5)

Inputs (per model_X):
  - effects_by_window.pkl : list over window sizes -> list over sequences -> array of window effects
  - frags_by_window.pkl   : same structure, but with k-mer strings for each window

For each selected window size and each window:
  - Take its effect (ΔRg, as produced by embedding_occlusion_effect)
  - Assign that effect to each residue in the fragment (k-mer)
  - Accumulate sum(effect) and count per amino acid

Output:
  - CSV with per-AA mean ΔRg
  - Bar plot of mean ΔRg per amino acid
"""

import argparse
from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        required=True,
        help="Root directory containing model_* subdirs (e.g. protbert_sliding_results_top5).",
    )
    ap.add_argument(
        "--windows",
        default="3,4,5",
        help="Comma-separated window sizes to include, e.g. '3,4,5' or '1,2,3,4,5,6,7,8,9,10'.",
    )
    ap.add_argument(
        "--out_csv",
        default="mean_delta_rg_by_residue.csv",
        help="Output CSV filename.",
    )
    ap.add_argument(
        "--out_png",
        default="mean_delta_rg_by_residue.png",
        help="Output PNG barplot filename.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    results_root = Path(args.results_root)
    requested_windows = [int(w.strip()) for w in args.windows.split(",")]

    # Find model_* directories
    model_dirs = sorted(
        [d for d in results_root.glob("model_*") if d.is_dir()],
        key=lambda p: p.name,
    )
    if not model_dirs:
        raise RuntimeError(f"No model_* directories found under {results_root}")

    print("Models found:", [d.name for d in model_dirs])
    print("Requested window sizes:", requested_windows)

    # Accumulators per amino acid
    sum_effect = defaultdict(float)
    count_effect = defaultdict(float)

    for mdir in model_dirs:
        print(f"\n=== Processing {mdir.name} ===")

        effects_path = mdir / "effects_by_window.pkl"
        frags_path = mdir / "frags_by_window.pkl"

        if not effects_path.exists() or not frags_path.exists():
            print(f"  [warn] Missing effects_by_window.pkl or frags_by_window.pkl in {mdir}; skipping.")
            continue

        with open(effects_path, "rb") as f:
            effects_by_window = pickle.load(f)
        with open(frags_path, "rb") as f:
            frags_by_window = pickle.load(f)

        n_windows = len(effects_by_window)
        # Window sizes are typically [1..n_windows], index = window_size - 1
        available_windows = list(range(1, n_windows + 1))
        print(f"  Available windows: {available_windows}")

        for w in requested_windows:
            if w < 1 or w > n_windows:
                print(f"  [warn] Window {w} not present in {mdir.name}; skipping.")
                continue

            w_idx = w - 1
            eff_list_per_seq = effects_by_window[w_idx]
            frag_list_per_seq = frags_by_window[w_idx]

            if len(eff_list_per_seq) != len(frag_list_per_seq):
                print(
                    f"  [warn] effects and frags length mismatch for window {w} "
                    f"in {mdir.name}; using min length."
                )

            n_seq = min(len(eff_list_per_seq), len(frag_list_per_seq))
            print(f"  -> window={w}, sequences={n_seq}")

            for eff_seq, frag_seq in zip(
                eff_list_per_seq[:n_seq], frag_list_per_seq[:n_seq]
            ):
                eff_seq = np.asarray(eff_seq, dtype=float)
                frag_seq = np.asarray(frag_seq, dtype=str)

                if len(eff_seq) != len(frag_seq):
                    L = min(len(eff_seq), len(frag_seq))
                    eff_seq = eff_seq[:L]
                    frag_seq = frag_seq[:L]

                # For each window: assign its effect to each residue in the fragment
                for effect, frag in zip(eff_seq, frag_seq):
                    frag = str(frag)
                    if not frag:
                        continue
                    for aa in frag:
                        if aa not in STANDARD_AA:
                            continue
                        sum_effect[aa] += effect
                        count_effect[aa] += 1.0

    if not sum_effect:
        raise RuntimeError("No effects accumulated; check inputs and window settings.")

    # Build DataFrame of mean ΔRg per residue
    aas = sorted(STANDARD_AA)
    mean_vals = []
    counts = []
    for aa in aas:
        c = count_effect.get(aa, 0.0)
        s = sum_effect.get(aa, 0.0)
        counts.append(c)
        mean_vals.append(s / c if c > 0 else 0.0)

    df = pd.DataFrame(
        {
            "AA": aas,
            "count": counts,
            "mean_delta_Rg": mean_vals,
        }
    )

    df.to_csv(args.out_csv, index=False)
    print(f"\n[saved] Mean ΔRg per residue → {args.out_csv}")

    # Make barplot
    plt.figure(figsize=(8, 4))
    plt.bar(df["AA"], df["mean_delta_Rg"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Amino acid")
    plt.ylabel("Mean ΔRg (window effect)")
    plt.title(
        f"Mean ΔRg per residue\n"
        f"(pooled across models {', '.join(d.name for d in model_dirs)}, "
        f"windows {requested_windows})"
    )
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()
    print(f"[saved] Barplot → {args.out_png}")


if __name__ == "__main__":
    main()
