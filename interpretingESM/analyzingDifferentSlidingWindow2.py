import pickle
import numpy as np
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl

sequences = []
with open('../training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
            print(" " in row[0], counter)
        counter += 1
with open("allEffects4.pkl", "rb") as f:
    allEffects = pickle.load(f)
with open("allFragments4.pkl", "rb") as f:
    allFragments = pickle.load(f)

print(len(allEffects))
print(len(allFragments))


# Normalizing sequences to smooth out any issues with the z-scores

def normalize_per_sequence(d):
    """
    Normalize a ΔRg array (per sequence) using z-score.
    Handles near-zero variance gracefully.
    """
    d = np.asarray(d)
    mean = d.mean()
    std  = d.std()

    # Avoid division by zero for flat sequences
    if std < 1e-12:
        return np.zeros_like(d)

    return (d - mean) / std
#getting residue level contributions at the sequence level
'''
def getResidueContributions(deltas, window):
    L = len(deltas) + window - 1
    res_scores = np.zeros(L)
    counts = np.zeros(L)
    for start, d in enumerate(deltas):
        end = start + window
        res_scores[start:end] += d
        counts[start:end] += 1
    return res_scores/counts


def getAllResiduesDeltas(sequences, deltaRgs, kValues):

    residue_delta = {}

    for ki, k in enumerate(kValues):
        residue_delta[k] = {}
        for s_idx, seq in enumerate(sequences):
            deltas = np.array(deltaRgs[ki][s_idx])
            residueLevel = getResidueContributions(deltas, k)
            residue_delta[k][s_idx] = residueLevel
    return residue_delta

def getPerResidueDataset(sequences, residueDelta):
    globalPairs = []
    for k, seqDict in residueDelta.items():
        for si, s in enumerate(sequences):
            R = seqDict[si]
            for i, aa in enumerate(s):
                globalPairs.append((aa, float(R[i])))
    return globalPairs

def aggregateResidues(globalPairs):
    values = defaultdict(list)
    for aa, d in globalPairs:
        values[aa].append(d)
    stats = {}
    for aa, arr in values.items():
        arr = np.array(arr)
        stats[aa] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "count": len(arr)
        }
    return stats

def getInfluences(sequences, deltaRgs, windows):
    residueDeltas = getAllResiduesDeltas(sequences, deltaRgs, windows)
    pairs = getPerResidueDataset(sequences, residueDeltas)
    finalStats = aggregateResidues(pairs)

    return finalStats, pairs, residueDeltas

'''

def getResidueContributions(delta_list, k, seq_len):
    assert len(delta_list) == seq_len - k + 1, \
    f"Mismatch: got {len(delta_list)} deltas but seq_len={seq_len}, k={k}"

    res_scores = np.zeros(seq_len)
    counts = np.zeros(seq_len)

    for start, d in enumerate(delta_list):
        end = start + k
        res_scores[start:end] += d
        counts[start:end] += 1

    # avoid division by zero
    counts[counts == 0] = 1  
    return res_scores / counts

def getAllResiduesDeltas(deltaRgs_for_seq, k_values, seq_len):
    per_k = []

    for k in k_values:
        delta_list = deltaRgs_for_seq[k]
        res_drg = getResidueContributions(delta_list, k, seq_len)
        res_drg = normalize_per_sequence(res_drg)
        per_k.append(res_drg)

    # simple equal-weight average across k
    per_k = np.array(per_k)    # shape (#k, L)
    return per_k.mean(axis=0)  # shape (L,)


def getPerResidueDataset(sequences, deltaRgs, k_values):
    global_pairs = []

    for s_idx, seq in enumerate(sequences):
        L = len(seq)
        
        # Build dict: k -> fragment ΔRg list for this seq
        deltas_for_seq = {k: deltaRgs[ki][s_idx] for ki, k in enumerate(k_values)}

        # Per-residue ΔRg averaged across k
        residue_drg = getAllResiduesDeltas(deltas_for_seq, k_values, L)

        # Pair residue with its ΔRg
        for aa, drg in zip(seq, residue_drg):
            global_pairs.append((aa, float(drg)))

    return global_pairs
def getInfluences(global_pairs, min_count = 30):
    values = defaultdict(list)
    for aa, drg in global_pairs:
        values[aa].append(drg)

    stats = {}
    for aa, arr in values.items():
        arr = np.array(arr)

        if len(arr) < min_count:
            continue  # drop rare residues

        stats[aa] = {
            'mean': arr.mean(),
            'std': arr.std(),
            'count': len(arr)
        }
    return stats


k_values = [1,2,3,4,5,6,7,8,9,10]   # whatever you use

pairs = getPerResidueDataset(sequences, allEffects, k_values)
stats = getInfluences(pairs)


# -----------------------------
# Global style (baseline-safe)
# -----------------------------
mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
})

# -----------------------------
# Data
# -----------------------------
aas = sorted(stats.keys())
means = np.array([stats[a]["mean"] for a in aas])

# -----------------------------
# Restrained sign encoding
# -----------------------------
pos_color = "#6F7F99"   # muted blue-gray (positive ΔRg)
neg_color = "#B7C0CC"   # lighter same-hue gray (negative ΔRg)

colors = []
alphas = []
for m in means:
    if m >= 0:
        colors.append(pos_color)
        alphas.append(0.75)
    else:
        colors.append(neg_color)
        alphas.append(0.85)

# -----------------------------
# Figure (single-column)
# -----------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.2))

for a, m, c, al in zip(aas, means, colors, alphas):
    ax.bar(
        a,
        m,
        width=0.6,
        color=c,
        alpha=al,
        edgecolor="none"
    )

# Zero as reference anchor (not decision boundary)
ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)

# -----------------------------
# Labels and title
# -----------------------------
ax.set_xlabel("Residue")
ax.set_ylabel("Mean ΔRg")
ax.set_title("Reference distribution of mean ΔRg")

# -----------------------------
# Axes cleanup
# -----------------------------
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="x", length=2)
ax.tick_params(axis="y", length=2)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))

plt.tight_layout()
plt.savefig("reference_mean_dRg_by_residue.pdf", bbox_inches="tight")
plt.show()

'''
mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
})

aas = sorted(stats.keys())
means = np.array([stats[a]["mean"] for a in aas])

# Sign-aware but restrained colors
pos_color = "#5B6F8E"   # darker blue-gray
neg_color = "#B7C0CC"   # lighter blue-gray
colors = np.where(means >= 0, pos_color, neg_color)

fig, ax = plt.subplots(figsize=(3.4, 2.2))  # single-column baseline

ax.bar(
    aas,
    means,
    width=0.6,
    color=colors,
    alpha=0.8,
    edgecolor="none"
)

# Zero as reference, not decision boundary
ax.axhline(0, color="black", linewidth=0.6, alpha=0.7)

ax.set_xlabel("Residue")
ax.set_ylabel("Mean ΔRg")
ax.set_title("Mean ΔRg by residue type")

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Quiet ticks
ax.tick_params(axis="x", length=2)
ax.tick_params(axis="y", length=2)

plt.tight_layout()
plt.savefig("baseline_mean_dRg_by_residue.pdf", bbox_inches="tight")
plt.show()
'''
'''
# --- Style setup (do this once per script) ---
mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
})

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
})

aas = sorted(stats.keys())
means = np.array([stats[a]["mean"] for a in aas])

fig, ax = plt.subplots(figsize=(3.4, 2.2))  # single-column baseline

ax.bar(
    aas,
    means,
    width=0.6,
    color="#7A7A7A",
    alpha=0.7,
    edgecolor="none"
)

ax.axhline(0, color="black", linewidth=0.6, alpha=0.7)

ax.set_xlabel("Residue")
ax.set_ylabel("Mean ΔRg")
ax.set_title("Mean ΔRg by residue type")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="x", length=2)
ax.tick_params(axis="y", length=2)

plt.tight_layout()
plt.savefig("baseline_mean_dRg_by_residue.pdf", bbox_inches="tight")
plt.show()
'''
'''
# Data
aas = sorted(stats.keys())
means = np.array([stats[a]['mean'] for a in aas])

# Colors: positive vs negative
colors = np.where(means >= 0, "#4C72B0", "#DD8452")

# Figure sized for single-column output
fig, ax = plt.subplots(figsize=(3.4, 2.2))

ax.bar(aas, means, color=colors, edgecolor="none")

# Zero reference line
ax.axhline(0, color="black", linewidth=0.8)

# Labels
ax.set_xlabel("Residue")
ax.set_ylabel("Mean ΔRg")

# Clean up spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Ticks
ax.tick_params(axis="x", rotation=0, length=3)
ax.tick_params(axis="y", length=3)

plt.tight_layout()
plt.savefig("mean_dRg_by_residue.pdf", bbox_inches="tight")
plt.show()
'''