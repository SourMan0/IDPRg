import numpy as np
from collections import defaultdict
import pandas as pd
import pickle
import matplotlib as mpl
import logomaker
import matplotlib.pyplot as plt

with open("allEffects4.pkl", "rb") as f:
    allEffects = pickle.load(f)
with open("allFragments4.pkl", "rb") as f:
    allFragments = pickle.load(f)


def zscoreNormalize(dRg_seq, eps = 1e-12, clip = None):
    dRg = np.asarray(dRg_seq, dtype = float)
    mu = dRg.mean()
    sigma = dRg.std()

    if sigma < eps:
        return np.zeros_like(dRg)
    else:
        z = (dRg - mu) / sigma 
    
    if clip is not None:
        z = np.clip(z, -clip, clip)
    w = -z
    return w
    
def computeWeightedResidueEffects(all_dRg, all_frags, k_vals, min_count = 20, use_abs_weight = False):
    results = {}
    for k in k_vals:
        k_idx = k - 1

        # Basically for each position in the motif, you want to have a dictionary of amino acids
        # with the relavant information
        sum_effect = [defaultdict(float) for _ in range(k)]
        count = [defaultdict(float) for _ in range(k)]

        # going over the sequences

        for dRgSeq, fragSeq in zip(all_dRg[k_idx], all_frags[k_idx]):
            dRgSeq = np.asarray(dRgSeq)

            dRg_norm = zscoreNormalize(dRgSeq)

            assert len(dRg_norm) == len(fragSeq)

            for dRg, frag in zip(dRg_norm, fragSeq):
                if abs(dRg) < 1e-6:
                    continue
                for pos, aa in enumerate(frag):
                    weight = abs(dRg) if use_abs_weight else 1.00
                    sum_effect[pos][aa] += dRg * weight
                    count[pos][aa] += weight
        mean_effect = []
        for pos in range(k):
            # Dictionary of amino acids for the mean effects at each position
            pos_dict = {}
            for aa, tot_w in sum_effect[pos].items():
                if count[pos][aa] < min_count:
                    pos_dict[aa] = 0.0
                else:
                    print(f"K: {k}, aa: {aa}, pos: {pos}, count: {count[pos][aa]}")
                    pos_dict[aa] = tot_w / count[pos][aa]
            mean_effect.append(pos_dict)
        results[k] = mean_effect
    return results

def effects_to_df(mean_effect):
    aas_set = set()
    for pos_dict in mean_effect:
        for aa in pos_dict.keys():
            aas_set.add(aa)
    aas = sorted(aas_set)
    data = []
    for pos_dict in mean_effect:
        row = [pos_dict.get(aa, 0.0) for aa in aas]
        data.append(row)
    df = pd.DataFrame(data, columns = aas)
    df.index = np.arange(1, len(mean_effect) + 1)
    return df


k_vals = list(range(1, 11))
results = computeWeightedResidueEffects(allEffects, allFragments, k_vals, min_count = 500)
df3 = effects_to_df(results[3])
df6 = effects_to_df(results[6])
df10  = effects_to_df(results[10])

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
})
fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize=(6.8, 4.8), sharex = "col", gridspec_kw={"wspace": 0.25, "hspace": 0.25})

dfs = {3: df3, 6: df6, 10: df10}
aa_colors = {
    # Hydrophobic (deep slate blue)
    'A': '#4C72B0', 'V': '#4C72B0', 'L': '#4C72B0',
    'I': '#4C72B0', 'M': '#4C72B0',

    # Aromatic (near-black charcoal)
    'F': '#2F2F2F', 'Y': '#2F2F2F', 'W': '#2F2F2F',

    # Polar uncharged (deep teal)
    'S': '#3C8D7D', 'T': '#3C8D7D', 'N': '#3C8D7D', 'Q': '#3C8D7D',

    # Positively charged (navy)
    'K': '#2B4F81', 'R': '#2B4F81', 'H': '#2B4F81',

    # Negatively charged (muted maroon)
    'D': '#8C3B3B', 'E': '#8C3B3B',

    # Special cases
    'P': '#6B8E23',   # olive (rigidity)
    'G': '#B58900',   # dark mustard
    'C': '#6E6E6E'    # neutral gray
}
for row, (frag_size, df) in enumerate(dfs.items()):
    df_expand = df.clip(lower = 0.0)
    axes[row, 0].set_xlim(0, frag_size)
    axes[row, 1].set_xlim(0, frag_size)
    logomaker.Logo(
        df_expand, ax = axes[row, 0],
        color_scheme = aa_colors, shade_below=0.6,
        fade_below=0.6, stack_order = 'small_on_top', baseline_width = 0
    )

    df_compact = (-df).clip(lower = 0.0)
    
    logomaker.Logo(
        df_compact, 
        ax=axes[row, 1],
        color_scheme=aa_colors,
        shade_below = 0.6,
        fade_below = 0.6,
        stack_order = "small_on_top",
        baseline_width = 0
    )
    axes[row, 0].set_ylabel(f"Size {frag_size}")

axes[0, 0].set_title("Expansion")
axes[0, 1].set_title("Compaction")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)
    ax.yaxis.set_ticks([])   # qualitative baseline

# X labels only on bottom row
axes[-1, 0].set_xlabel("Position in motif")
axes[-1, 1].set_xlabel("Position in motif")

plt.tight_layout()
plt.savefig("baseline_logos_fragment_sizes.pdf", bbox_inches="tight")
plt.savefig("baseline_logos_fragment_sizes.png", bbox_inches="tight", dpi=300)
plt.show()

'''
df_expand = df.clip(lower = 0.0)
df_compact = (-df).clip(lower = 0.0)


plt.figure(figsize=(12, 4))
logomaker.Logo(df_expand, color_scheme = 'chemistry')    
plt.xlabel("Position in Motif")
plt.title("Expansion Logo")
plt.ylabel("Enrichment (scaled)")
plt.tight_layout()
plt.show()
logomaker.Logo(df_compact, color_scheme = 'chemistry')
plt.title("Compaction Logo")
plt.show()
'''