# All-amino-acid sequence logos.
# -------------------------------
# Same z-score-weighted residue-effects computation as findMotifs3.py, but with
# two differences for the "all amino acids represented" variant:
#
#   1. min_count is dropped from 500 to 20 so every amino acid that's been
#      observed at least a handful of times at a position shows up. (The
#      journal logo at min_count=500 deliberately drops rare residues; this
#      figure keeps everything.)
#   2. The colour palette is logomaker's built-in 'chemistry' scheme, which
#      assigns a distinct colour to each of the 20 standard amino acids
#      (rather than the chemistry-group palette in findMotifs3.py that maps
#      multiple AAs to the same colour). So the reader can distinguish every
#      residue at every position.
#
# Reads the leak-free occlusion pickles regenerated from krr_pipeline.joblib
# (ESM-6 layer 1, PCA=100, KernelRidge) on May 31.
#
# Output: baseline_sequence_logos_all_aas.{pdf,png}

import pickle
from collections import defaultdict

import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


with open("allEffects4.pkl", "rb") as f:
    allEffects = pickle.load(f)
with open("allFragments4.pkl", "rb") as f:
    allFragments = pickle.load(f)


def zscoreNormalize(dRg_seq, eps=1e-12, clip=None):
    dRg = np.asarray(dRg_seq, dtype=float)
    mu = dRg.mean()
    sigma = dRg.std()
    if sigma < eps:
        return np.zeros_like(dRg)
    z = (dRg - mu) / sigma
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return -z  # negate so positive z = compaction-driving


def computeWeightedResidueEffects(all_dRg, all_frags, k_vals,
                                  min_count=20, use_abs_weight=False):
    """Mean signed z-scored ΔRg per (position, amino acid).

    The min_count threshold zeros out (position, AA) pairs we've barely seen,
    to keep numerical noise from being plotted as a real signal. The default
    here is 20 -- low enough that essentially every observed AA at every
    position survives, which is what "all amino acids represented" means.
    """
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
                    w = abs(dRg) if use_abs_weight else 1.00
                    sum_effect[pos][aa] += dRg * w
                    count[pos][aa] += w
        mean_effect = []
        for pos in range(k):
            pos_dict = {}
            for aa, tot_w in sum_effect[pos].items():
                pos_dict[aa] = 0.0 if count[pos][aa] < min_count else tot_w / count[pos][aa]
            mean_effect.append(pos_dict)
        results[k] = mean_effect
    return results


def effects_to_df(mean_effect):
    aas_set = set()
    for pos_dict in mean_effect:
        for aa in pos_dict.keys():
            aas_set.add(aa)
    aas = sorted(aas_set)
    data = [[pos_dict.get(aa, 0.0) for aa in aas] for pos_dict in mean_effect]
    df = pd.DataFrame(data, columns=aas)
    df.index = np.arange(1, len(mean_effect) + 1)
    return df


k_vals = list(range(1, 11))
results = computeWeightedResidueEffects(allEffects, allFragments, k_vals, min_count=20)
dfs = {k: effects_to_df(results[k]) for k in (3, 6, 10)}

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
})

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6.8, 4.8),
                          sharex="col",
                          gridspec_kw={"wspace": 0.25, "hspace": 0.25})

for row, (k, df) in enumerate(dfs.items()):
    df_expand = df.clip(lower=0.0)
    df_compact = (-df).clip(lower=0.0)

    axes[row, 0].set_xlim(0, k)
    axes[row, 1].set_xlim(0, k)

    logomaker.Logo(df_expand, ax=axes[row, 0],
                   color_scheme='chemistry',
                   shade_below=0.6, fade_below=0.6,
                   stack_order='small_on_top',
                   baseline_width=0)
    logomaker.Logo(df_compact, ax=axes[row, 1],
                   color_scheme='chemistry',
                   shade_below=0.6, fade_below=0.6,
                   stack_order='small_on_top',
                   baseline_width=0)

    axes[row, 0].set_ylabel(f"Size {k}")

axes[0, 0].set_title("Expansion")
axes[0, 1].set_title("Compaction")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)
    ax.yaxis.set_ticks([])

axes[-1, 0].set_xlabel("Position in motif")
axes[-1, 1].set_xlabel("Position in motif")

plt.tight_layout()
plt.savefig("baseline_sequence_logos_all_aas.pdf", bbox_inches="tight")
plt.savefig("baseline_sequence_logos_all_aas.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("-> baseline_sequence_logos_all_aas.pdf (+ .png)")
