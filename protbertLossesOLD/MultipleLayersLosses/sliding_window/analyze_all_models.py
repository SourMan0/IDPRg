#!/usr/bin/env python3
# analyze_all_models.py
"""
Batch motif/enrichment analysis for all ProtBERT sliding-window models.

Given a results root directory like:
    protbert_sliding_results_all_only/
        model_1/
            effects_by_window.pkl
            frags_by_window.pkl
        model_2/
        ...

This script loops through every model_* folder, performs the same
enrichment analysis used in analyze_sliding_results.py, and SAVES PNG
files + a summary CSV.

Example:
    python3 analyze_all_models.py \
        --results_root protbert_sliding_results_all_only \
        --window 5 \
        --topk 5000
"""

import argparse, os, pickle
import numpy as np
import matplotlib.pyplot as plt

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
aa_to_idx = {a:i for i,a in enumerate(AA)}

def weighted_profile(seqs, weights):
    """Compute position-by-AA weighted frequency profile."""
    k = len(seqs[0])
    prof = np.zeros((k, 20))
    for s, w in zip(seqs, weights):
        for i, ch in enumerate(s):
            if ch in aa_to_idx:
                prof[i, aa_to_idx[ch]] += w
    prof /= np.sum(prof) + 1e-9
    return prof


def analyze_one_model(results_dir, window=5, topk=5000):
    """Load sliding window results for one model and compute enrichment."""
    effects_by_window = pickle.load(open(os.path.join(results_dir, "effects_by_window.pkl"), "rb"))
    frags_by_window   = pickle.load(open(os.path.join(results_dir, "frags_by_window.pkl"), "rb"))

    w_idx = window - 1
    all_effects = []
    all_frags = []

    # Collect effects and fragments for this window size
    for effects_seq, frags_seq in zip(effects_by_window[w_idx], frags_by_window[w_idx]):
        effects_seq = np.array(effects_seq)
        frags_seq = np.array(frags_seq, dtype=str)

        pad = window // 2
        centered = effects_seq[pad : len(effects_seq) - pad]

        all_effects.append(centered)
        all_frags.append(frags_seq)

    all_effects = np.concatenate(all_effects)
    all_frags = np.concatenate(all_frags)

    # Top-k fragment selection
    idx = np.argsort(np.abs(all_effects))[::-1][:topk]
    frags = all_frags[idx]
    effs  = all_effects[idx]

    pos_mask = effs > 0
    neg_mask = effs < 0

    pos_frags, pos_w = frags[pos_mask], effs[pos_mask]
    neg_frags, neg_w = frags[neg_mask], np.abs(effs[neg_mask])

    pos_prof = weighted_profile(pos_frags, pos_w)
    neg_prof = weighted_profile(neg_frags, neg_w)

    enrichment = np.log2((pos_prof + 1e-6) / (neg_prof + 1e-6))

    stats = {
        "results_dir": results_dir,
        "topk": len(frags),
        "pos_count": int(len(pos_frags)),
        "neg_count": int(len(neg_frags)),
        "pos_frac": float(len(pos_frags) / max(len(frags), 1)),
        "neg_frac": float(len(neg_frags) / max(len(frags), 1)),
        "mean_abs_effect": float(np.mean(np.abs(effs))) if len(effs) else 0.0,
        "max_abs_effect": float(np.max(np.abs(effs))) if len(effs) else 0.0,
    }

    return enrichment, stats


def save_heatmap(enrichment, window, out_png, title):
    """Save a log2-enrichment heatmap to disk."""
    plt.figure(figsize=(9, 7))
    im = plt.imshow(enrichment.T, aspect="auto", origin="lower", vmin=-2, vmax=2)
    plt.colorbar(im, label="log2 enrichment (Rg↑ / Rg↓)")
    plt.xticks(range(window), [f"Pos{i+1}" for i in range(window)])
    plt.yticks(range(20), AA)
    plt.xlabel("Position in motif")
    plt.ylabel("Amino acid")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="Directory with model_* subfolders containing sliding-window results.")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--out_dir", default=None,
                    help="Where to save output. Default: <results_root>/analysis_window{window}")
    args = ap.parse_args()

    results_root = args.results_root
    out_dir = args.out_dir or os.path.join(results_root, f"analysis_window{args.window}")
    os.makedirs(out_dir, exist_ok=True)

    model_dirs = sorted(
        os.path.join(results_root, d)
        for d in os.listdir(results_root)
        if d.startswith("model_") and os.path.isdir(os.path.join(results_root, d))
    )

    if not model_dirs:
        raise RuntimeError(f"No model_* folders found in {results_root}")

    summary_rows = []

    for mdir in model_dirs:
        model_name = os.path.basename(mdir)
        print(f"[processing] {model_name}...")

        try:
            enrichment, stats = analyze_one_model(mdir, window=args.window, topk=args.topk)
        except FileNotFoundError:
            print(f"[skip] {model_name} missing sliding-window pickles")
            continue

        out_png = os.path.join(out_dir, f"{model_name}_enrichment_w{args.window}.png")
        title = f"{model_name} (window={args.window})"
        save_heatmap(enrichment, args.window, out_png, title)

        stats["model"] = model_name
        stats["window"] = args.window
        stats["out_png"] = out_png
        summary_rows.append(stats)

        print(f"[done] saved {out_png}")

    # Save summary CSV
    try:
        import pandas as pd
        summary_csv = os.path.join(out_dir, f"summary_w{args.window}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"[saved] summary -> {summary_csv}")
    except Exception as e:
        print(f"[warn] could not save summary CSV: {e}")

    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()