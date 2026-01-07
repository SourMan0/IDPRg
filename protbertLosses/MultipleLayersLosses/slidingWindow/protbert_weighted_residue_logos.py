#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import logomaker


# ---------- 1. Your original helper functions (unchanged) ----------

def zscoreNormalize(dRg_seq, eps=1e-12, clip=None):
    dRg = np.asarray(dRg_seq, dtype=float)
    mu = dRg.mean()
    sigma = dRg.std()

    if sigma < eps:
        return np.zeros_like(dRg)
    else:
        z = (dRg - mu) / (sigma + eps)

    if clip is not None:
        z = np.clip(z, -clip, clip)

    w = -z  # same sign convention as your ESM code
    return w


def computeWeightedResidueEffects(all_dRg, all_frags, k_vals, min_count=20, use_abs_weight=False):
    results = {}
    for k in k_vals:
        k_idx = k - 1

        sum_effect = [defaultdict(float) for _ in range(k)]
        count = [defaultdict(float) for _ in range(k)]

        for dRgSeq, fragSeq in zip(all_dRg[k_idx], all_frags[k_idx]):
            dRgSeq = np.asarray(dRgSeq)
            dRg_norm = zscoreNormalize(dRgSeq)

            assert len(dRg_norm) == len(fragSeq)

            for dRg, frag in zip(dRg_norm, fragSeq):
                if abs(dRg) < 1e-6:
                    continue
                for pos, aa in enumerate(frag):
                    weight = abs(dRg) if use_abs_weight else 1.0
                    sum_effect[pos][aa] += dRg * weight
                    count[pos][aa] += weight

        mean_effect = []
        for pos in range(k):
            pos_dict = {}
            for aa, tot_w in sum_effect[pos].items():
                if count[pos][aa] < min_count:
                    pos_dict[aa] = 0.0
                else:
                    pos_dict[aa] = tot_w / count[pos][aa]
            mean_effect.append(pos_dict)

        results[k] = mean_effect
    return results


def effects_to_df(mean_effect):
    aas_set = set()
    for pos_dict in mean_effect:
        aas_set.update(pos_dict.keys())
    aas = sorted(aas_set)

    data = []
    for pos_dict in mean_effect:
        row = [pos_dict.get(aa, 0.0) for aa in aas]
        data.append(row)

    df = pd.DataFrame(data, columns=aas)
    df.index = np.arange(1, len(mean_effect) + 1)
    return df


# ---------- 2. Build allEffects / allFragments from ProtBERT results ----------

def build_all_effects_from_protbert(results_root: str):
    """
    results_root: e.g. 'protbert_sliding_results_top5'

    Returns:
        allEffects, allFragments

    where:
        allEffects[k_idx]   = list over sequences (across ALL models)
        allFragments[k_idx] = same shape, but k-mer strings
    """
    root = Path(results_root)
    model_dirs = sorted([d for d in root.glob("model_*") if d.is_dir()],
                        key=lambda p: p.name)

    if not model_dirs:
        raise RuntimeError(f"No model_* dirs found under {results_root}")

    # Infer number of window sizes from first model
    with open(model_dirs[0] / "effects_by_window.pkl", "rb") as f:
        eff0 = pickle.load(f)
    with open(model_dirs[0] / "frags_by_window.pkl", "rb") as f:
        frag0 = pickle.load(f)

    n_windows = len(eff0)
    allEffects = [[] for _ in range(n_windows)]
    allFragments = [[] for _ in range(n_windows)]

    for mdir in model_dirs:
        print(f"Loading sliding results from {mdir.name} ...")
        with open(mdir / "effects_by_window.pkl", "rb") as f:
            eff = pickle.load(f)
        with open(mdir / "frags_by_window.pkl", "rb") as f:
            frags = pickle.load(f)

        if len(eff) != n_windows or len(frags) != n_windows:
            raise ValueError(f"Window count mismatch in {mdir}")

        # Append sequences across models for each window size
        for k_idx in range(n_windows):
            allEffects[k_idx].extend(eff[k_idx])      # list of seqs
            allFragments[k_idx].extend(frags[k_idx])  # list of seqs

    return allEffects, allFragments


# ---------- 3. Main: compute logos for a chosen window size ----------

def main():
    results_root = "protbert_sliding_results_top5"
    k_vals = list(range(1, 11))   # assuming windows 1..10 like your setup
    target_k = 5               # choose which window size to visualize

    allEffects, allFragments = build_all_effects_from_protbert(results_root)

    results = computeWeightedResidueEffects(
        allEffects,
        allFragments,
        k_vals,
        min_count=500,           # same as your ESM code
        use_abs_weight=False
    )

    mean_effect_k = results[target_k]
    df = effects_to_df(mean_effect_k)

    # Separate expansion vs compaction, same as ESM code
    df_expand = df.clip(lower=0.0)
    df_compact = (-df).clip(lower=0.0)

    # Plot expansion logo
    plt.figure(figsize=(12, 4))
    logomaker.Logo(df_expand, color_scheme="chemistry")
    plt.xlabel("Position in Motif")
    plt.ylabel("Enrichment (scaled)")
    plt.title(f"ProtBERT: residues increasing normalized effect (k={target_k})")
    plt.tight_layout()
    plt.savefig(f"protbert_k{target_k}_expand_logo.png", dpi=300)
    plt.close()

    # Plot compaction logo
    plt.figure(figsize=(12, 4))
    logomaker.Logo(df_compact, color_scheme="chemistry")
    plt.xlabel("Position in Motif")
    plt.ylabel("Enrichment (scaled)")
    plt.title(f"ProtBERT: residues decreasing normalized effect (k={target_k})")
    plt.tight_layout()
    plt.savefig(f"protbert_k{target_k}_compact_logo.png", dpi=300)
    plt.close()

    print(f"[saved] Logos for k={target_k} → protbert_k{target_k}_*_logo.png")


if __name__ == "__main__":
    main()