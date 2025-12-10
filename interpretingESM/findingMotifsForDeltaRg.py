import numpy as np
from collections import Counter, defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import pickle

with open("allEffects3.pkl", "rb") as f:
    allEffects = pickle.load(f)
with open("allFragments3.pkl", "rb") as f:
    allFragments = pickle.load(f)



# Counts number of amino acids per position
def position_counts(motifs):
    if len(motifs) == 0:
        return []

    k = len(motifs[0])
    pos_counts = [Counter() for _ in range(k)]

    for frag in motifs:
        for i, aa in enumerate(frag):
            pos_counts[i][aa] += 1

    return pos_counts

def get_background_counts_for_k(all_frags_for_k):
    flat = []
    for frag_list in all_frags_for_k:
        flat.extend(frag_list)
    return position_counts(flat)

# Normalizes the delta Rgs based on which sequence it's in. Since some sequences may have bigger changes based on 
# which protein it is, we want the *relative* change in Rg
def normalize_dRg(dRg):
    dRg = np.asarray(dRg)
    hi = np.percentile(np.abs(dRg), 95)
    if hi < 1e-12:
        return np.zeros_like(dRg)
    return dRg / hi
'''
def normalize_dRg(dRg):
    dRg = np.asarray(dRg)
    maxval = np.max(np.abs(dRg))
    if maxval < 1e-12:
        return np.zeros_like(dRg)
    return dRg / maxval
'''
# This function extracts the normalized sequences motifs, looking at the top/bottom percentage which will determine
# whether they are expanding or contracting

def extract_motifs_normalized(all_dRg, all_frags, k_values, top_pct=0.10, bot_pct=0.10):
    expanding = {k: [] for k in k_values}
    compacting = {k: [] for k in k_values}
    for k in k_values:
        pooled_dRg = []
        pooled_frags = []
        for dRg_seq, frag_seq in zip(all_dRg[k - 1], all_frags[k - 1]):
            
            dRg_norm = normalize_dRg(dRg_seq).tolist()

            pooled_dRg.extend(dRg_norm)
            pooled_frags.extend(frag_seq)

        hi = np.percentile(pooled_dRg, 100*(1-top_pct))
        lo = np.percentile(pooled_dRg, 100*bot_pct)

        pooled_frags = np.array(pooled_frags)
        pooled_dRg = np.array(pooled_dRg)

        comp_mask  = pooled_dRg >= hi
        exp_mask = pooled_dRg <= lo

        expanding[k]  = list(zip(pooled_frags[exp_mask],  pooled_dRg[exp_mask]))
        compacting[k] = list(zip(pooled_frags[comp_mask], pooled_dRg[comp_mask]))
            
    return expanding, compacting

def compute_diff_enrichment(exp_counts, comp_counts, min_total=20, compOrExp = 'exp'):
    enrich = []
    for pos in range(len(exp_counts)):
        exp = exp_counts[pos]
        comp = comp_counts[pos]

        total_exp  = sum(exp.values())
        total_comp = sum(comp.values())

        all_aas = sorted(set(exp.keys()) | set(comp.keys()))
        pos_dict = {}

        for aa in all_aas:
            e = exp.get(aa, 0)
            c = comp.get(aa, 0)

            if e + c < min_total:
                pos_dict[aa] = 0.0
                continue

            f_exp  = e / total_exp
            f_comp = c / total_comp
            if compOrExp == 'exp':
                pos_dict[aa] = f_exp - f_comp
            else:
                pos_dict[aa] = f_comp - f_exp   # absolute difference

        enrich.append(pos_dict)

    return enrich

def compute_log_enrichment(exp_counts,
                           comp_counts,
                           pseudocount=1.0,
                           min_total=20):
    """
    Compute log2 enrichment of amino acids in expanding vs compacting motifs.

    Parameters
    ----------
    exp_counts : list of Counter
        exp_counts[pos][aa] = count of 'aa' at position 'pos' in expanding motifs.
    comp_counts : list of Counter
        comp_counts[pos][aa] = count of 'aa' at position 'pos' in compacting motifs.
    pseudocount : float
        Smoothing added to each amino-acid count (Dirichlet prior) to avoid
        log(0) and stabilize low counts.
    min_total : int
        Minimum total (exp + comp) count for an amino acid at a position to be
        considered. Below this, enrichment is set to 0 (treated as noise).

    Returns
    -------
    enrich : list of dict
        enrich[pos][aa] = log2( f_exp(aa,pos) / f_comp(aa,pos) ).
        Positive → enriched in expanding motifs.
        Negative → enriched in compacting motifs.
    """

    enrich = []  # will hold one dict per position

    # Loop over positions in the motif (0..k-1)
    for pos in range(len(exp_counts)):
        exp = exp_counts[pos]   # Counter for this position in expanding motifs
        comp = comp_counts[pos] # Counter for this position in compacting motifs

        # Total counts at this position in each group
        total_exp  = sum(exp.values())
        total_comp = sum(comp.values())

        # Union of all amino acids seen in expanding or compacting at this position
        all_aas = sorted(set(exp.keys()) | set(comp.keys()))
        K = len(all_aas)  # number of distinct residue types at this position

        pos_dict = {}  # enrichment scores for this position

        for aa in all_aas:
            # Raw counts for this amino acid
            e = exp.get(aa, 0)
            c = comp.get(aa, 0)

            # If we barely see this residue at this position, treat as noise
            if e + c < min_total:
                pos_dict[aa] = 0.0
                continue

            # Smoothed frequencies (Dirichlet smoothing with pseudocount)
            f_exp  = (e + pseudocount) / (total_exp  + pseudocount * K)
            f_comp = (c + pseudocount) / (total_comp + pseudocount * K)

            # Log2 enrichment: positive → enriched in expanding vs compacting
            pos_dict[aa] = np.log2(f_exp / f_comp)

        enrich.append(pos_dict)

    return enrich

'''
def compute_log_enrichment(exp_counts, comp_counts, pseudocount=1, min_total = 20):
    enrich = []

    for pos in range(len(exp_counts)):
        exp = exp_counts[pos]
        comp = comp_counts[pos]

        pos_dict = {}
        all_aas = set(exp.keys()) | set(comp.keys())

        for aa in all_aas:
            e = exp.get(aa, 0)
            c = comp.get(aa, 0)
            pos_dict[aa] = np.log2((e + pseudocount) / (c + pseudocount))

        enrich.append(pos_dict)

    return enrich
'''
def context_flips(motifs_pos, motifs_neg):
    flips = defaultdict(lambda: {"exp": Counter(), "comp": Counter()})

    def neighbors(frag):
        k = len(frag)
        for i in range(k):
            center = frag[i]
            left = frag[i-1] if i > 0 else None
            right = frag[i+1] if i < k-1 else None
            yield center, left, right

    for frag, d in motifs_pos:
        for center, left, right in neighbors(frag):
            if left: flips[center]["exp"][left] += 1
            if right: flips[center]["exp"][right] += 1

    for frag, d in motifs_neg:
        for center, left, right in neighbors(frag):
            if left: flips[center]["comp"][left] += 1
            if right: flips[center]["comp"][right] += 1

    return flips

def plot_enrichment_heatmap(enrich, title="Motif Enrichment Heatmap"):
    """
    enrich = list of dicts, one per position
             enrich[pos][aa] = log2 enrichment value
    """
    # convert to DataFrame (AA × position)
    aas = sorted({aa for pos in enrich for aa in pos.keys()})
    df = pd.DataFrame(index=aas)

    for i, pos_dict in enumerate(enrich):
        col = f"pos_{i+1}"
        df[col] = [pos_dict.get(aa, 0.0) for aa in df.index]

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.xlabel("Motif Position")
    plt.ylabel("Residue")
    plt.tight_layout()
    plt.show()
def plot_sequence_logo_clean(enrich, title="Motif Enrichment Logo", top_n=5):
    aas = sorted({aa for pos in enrich for aa in pos.keys()})
    df = pd.DataFrame(index=aas)

    # Build DF
    for i, pos_dict in enumerate(enrich):
        df[f"pos_{i+1}"] = [pos_dict.get(aa, 0.0) for aa in df.index]

    df2 = df.transpose()
    df2.index = np.arange(1, df2.shape[0] + 1)

    # --- CLEANING OPTIONS ---
    # 1. Convert log2 enrichment → probabilities
    df_clean = np.exp(df2)
    df_clean = df_clean.div(df_clean.sum(axis=1), axis=0)

    # 2. Keep only top-N residues (optional)
    if top_n is not None:
        for idx in df_clean.index:
            row = df_clean.loc[idx]
            top = row.nlargest(top_n).index
            df_clean.loc[idx] = [row[a] if a in top else 0 for a in df_clean.columns]

    # 3. Drop rarely used residues (e.g., U)
    drop_cols = [aa for aa in df_clean.columns if df_clean[aa].sum() == 0]
    df_clean = df_clean.drop(columns=drop_cols)

    # --- Plotting ---
    plt.figure(figsize=(12, 4))
    logomaker.Logo(df_clean,
                   color_scheme='chemistry')
    plt.title(title)
    plt.xlabel("Position in Motif")
    plt.ylabel("Enrichment (scaled)")
    plt.tight_layout()
    plt.show()
def plot_top_motifs(expanding, compacting, k, top_n=15):
    exp_sorted  = sorted(expanding[k], key=lambda x: x[1], reverse=True)[:top_n]
    comp_sorted = sorted(compacting[k], key=lambda x: x[1])[:top_n]

    exp_frags  = [f for f, d in exp_sorted]
    exp_vals   = [d for f, d in exp_sorted]
    comp_frags = [f for f, d in comp_sorted]
    comp_vals  = [d for f, d in comp_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].barh(exp_frags, exp_vals, color="red")
    axes[0].set_title(f"Top Expanding Motifs (k={k})")
    axes[0].invert_yaxis()

    axes[1].barh(comp_frags, comp_vals, color="blue")
    axes[1].set_title(f"Top Compacting Motifs (k={k})")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()
def plot_context_flip(flip, aa="I", top_n=10):
    """
    flip[aa] = { 'exp': Counter(), 'comp': Counter() }
    """
    exp = flip[aa]["exp"].most_common(top_n)
    comp = flip[aa]["comp"].most_common(top_n)

    neighbors_exp  = [x[0] for x in exp]
    counts_exp     = [x[1] for x in exp]
    neighbors_comp = [x[0] for x in comp]
    counts_comp    = [x[1] for x in comp]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].barh(neighbors_exp, counts_exp, color="red")
    ax[0].invert_yaxis()
    ax[0].set_title(f"Neighbors of {aa} in *Expanding* Motifs")

    ax[1].barh(neighbors_comp, counts_comp, color="blue")
    ax[1].invert_yaxis()
    ax[1].set_title(f"Neighbors of {aa} in *Compacting* Motifs")

    plt.tight_layout()
    plt.show()


expanding, compacting = extract_motifs_normalized(allEffects, allFragments, k_values=[3,5,7])

k = 5  # choose motif length

# MSA-style counts and enrichment
exp_counts = position_counts([f for f, d in expanding[k]])

comp_counts = position_counts([f for f, d in compacting[k]])

k = len(exp_counts)  # motif length, e.g. 5

for pos in range(k):
    print(f"\nPosition {pos+1}")
    print("expanding counts:", exp_counts[pos].most_common())
    print("compacting counts:", comp_counts[pos].most_common())

    eC = exp_counts[pos].get('C', 0)
    cC = comp_counts[pos].get('C', 0)
    print(f"  C: exp={eC}, comp={cC}, total={eC + cC}")
enrich = compute_diff_enrichment(exp_counts, comp_counts, compOrExp= 'comp')

# Visuals
plot_enrichment_heatmap(enrich, title=f"Motif Enrichment Heatmap (k={k})")
plot_sequence_logo_clean(enrich, title=f"Motif Enrichment Logo (k={k})")
plot_top_motifs(expanding, compacting, k)
flip = context_flips(expanding[k], compacting[k])
plot_context_flip(flip, aa="I")