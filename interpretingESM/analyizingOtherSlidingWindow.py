import pickle
import numpy as np
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

sequences = []
with open('training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
            print(" " in row[0], counter)
        counter += 1
with open("interpretingESM/allEffects2.pkl", "rb") as f:
    allEffects = pickle.load(f)
with open("interpretingESM/allFragments2.pkl", "rb") as f:
    allFragments = pickle.load(f)

print(len(allEffects))
print(len(allFragments))


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


allEffects2 = []
for k in allEffects:
    allEffects2.append([])
    for s in k:
        allEffects2[-1].append(normalize_per_sequence(s))

pairs = getPerResidueDataset(sequences, allEffects2, k_values)
stats = getInfluences(pairs)

# Plot
aas = sorted(stats.keys())
means = [stats[a]['mean'] for a in aas]


plt.figure(figsize=(10,4))
plt.bar(aas, means)
plt.axhline(0, color='black', linewidth=1)
plt.title("Mean ΔRg by Residue Type (Corrected Aggregation)")
plt.xlabel("Residue")
plt.ylabel("Mean ΔRg")
plt.tight_layout()
plt.show()