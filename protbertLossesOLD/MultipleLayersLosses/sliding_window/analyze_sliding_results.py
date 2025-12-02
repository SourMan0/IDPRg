#!/usr/bin/env python3
# analyze_sliding_results.py

import argparse, os, pickle
import numpy as np
import matplotlib.pyplot as plt

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
aa_to_idx = {a:i for i,a in enumerate(AA)}

def weighted_profile(seqs, weights):
    k = len(seqs[0])
    prof = np.zeros((k, 20))
    for s, w in zip(seqs, weights):
        for i, ch in enumerate(s):
            if ch in aa_to_idx:
                prof[i, aa_to_idx[ch]] += w
    prof /= np.sum(prof) + 1e-9
    return prof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="e.g., protbert_sliding_results_all_only/model_1")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--topk", type=int, default=5000)
    args = ap.parse_args()

    effects_by_window = pickle.load(open(os.path.join(args.results_dir, "effects_by_window.pkl"), "rb"))
    frags_by_window   = pickle.load(open(os.path.join(args.results_dir, "frags_by_window.pkl"), "rb"))

    w_idx = args.window - 1
    all_effects = []
    all_frags = []

    for effects_seq, frags_seq in zip(effects_by_window[w_idx], frags_by_window[w_idx]):
        effects_seq = np.array(effects_seq)
        frags_seq = np.array(frags_seq, dtype=str)

        # effects are centered; align with frags length
        pad = args.window // 2
        centered = effects_seq[pad: len(effects_seq)-pad]

        all_effects.append(centered)
        all_frags.append(frags_seq)

    all_effects = np.concatenate(all_effects)
    all_frags = np.concatenate(all_frags)

    # pick top-k by absolute magnitude
    idx = np.argsort(np.abs(all_effects))[::-1][:args.topk]
    frags = all_frags[idx]
    effs  = all_effects[idx]

    pos_mask = effs > 0
    neg_mask = effs < 0

    pos_frags, pos_w = frags[pos_mask], effs[pos_mask]
    neg_frags, neg_w = frags[neg_mask], np.abs(effs[neg_mask])

    print(f"TopK fragments: {len(frags)}")
    print(f"Positive (Rg↑): {len(pos_frags)}   Negative (Rg↓): {len(neg_frags)}")

    pos_prof = weighted_profile(pos_frags, pos_w)
    neg_prof = weighted_profile(neg_frags, neg_w)

    enrichment = np.log2((pos_prof + 1e-6) / (neg_prof + 1e-6))

    plt.figure(figsize=(9,7))
    im = plt.imshow(enrichment.T, aspect="auto", origin="lower", vmin=-2, vmax=2)
    plt.colorbar(im, label="log2 enrichment (Rg↑ / Rg↓)")
    plt.xticks(range(args.window), [f"Pos{i+1}" for i in range(args.window)])
    plt.yticks(range(20), AA)
    plt.xlabel("Position in motif")
    plt.ylabel("Amino acid")
    plt.title(f"ProtBERT sliding-window enrichment (window={args.window})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()