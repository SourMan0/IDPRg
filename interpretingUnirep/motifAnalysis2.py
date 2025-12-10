import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from collections import defaultdict

# ---------------------------------------------------------
# 1. Load window omissions from CSV
# ---------------------------------------------------------

def load_windows(csv_path, k=None):
    """
    Load (fragment, delta_rg) pairs from window_omissions_all.csv.

    Assumes header: omitted_seq, delta_rg, start_pos

    If k is not None, only keep fragments of length k.
    """
    frags = []
    dRg   = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frag = row["omitted_seq"].strip()
            drg  = float(row["delta_rg"])
            if (k is not None) and (len(frag) != k):
                continue
            frags.append(frag)
            dRg.append(drg)

    return np.array(frags, dtype=object), np.array(dRg, dtype=float)


# ---------------------------------------------------------
# 2. Normalization and motif-weight aggregation
#    (adapted from findingMotifs2.py)
# ---------------------------------------------------------

def zscoreNormalize(dRg_seq, eps=1e-12, clip=None):
    """
    z-score normalization with optional clipping, and sign flip
    (same as your labmate's code: w = -z).
    """
    dRg = np.asarray(dRg_seq, dtype=float)
    mu  = dRg.mean()
    sigma = dRg.std()

    if sigma < eps:
        z = np.zeros_like(dRg)
    else:
        z = (dRg - mu) / sigma

    if clip is not None:
        z = np.clip(z, -clip, clip)

    w = -z         # sign flip as in original code
    return w


def computeWeightedResidueEffects_from_csv(
    csv_path,
    k,
    min_count=20,
    use_abs_weight=False,
    clip_z=None,
):
    """
    Compute mean effect per (position, residue) for motifs of length k
    using window_omissions_all.csv.

    Returns
    -------
    mean_effect : list of dict
        mean_effect[pos][aa] = effect value (can be positive or negative).
    """
    frags, dRg = load_windows(csv_path, k=k)

    if len(frags) == 0:
        raise ValueError(f"No fragments of length k={k} found in {csv_path}")

    # Global normalization across all motifs of this k
    dRg_norm = zscoreNormalize(dRg, clip=clip_z)

    # sum_effect[pos][aa], count[pos][aa]
    sum_effect = [defaultdict(float) for _ in range(k)]
    count      = [defaultdict(float) for _ in range(k)]

    for drg, frag in zip(dRg_norm, frags):
        if abs(drg) < 1e-6:
            continue
        for pos, aa in enumerate(frag):
            weight = abs(drg) if use_abs_weight else 1.0
            sum_effect[pos][aa] += drg * weight
            count[pos][aa]      += weight

    mean_effect = []
    for pos in range(k):
        pos_dict = {}
        for aa, tot_w in sum_effect[pos].items():
            if count[pos][aa] < min_count:
                pos_dict[aa] = 0.0
            else:
                pos_dict[aa] = tot_w / count[pos][aa]
        mean_effect.append(pos_dict)

    return mean_effect


def effects_to_df(mean_effect):
    """
    Convert mean_effect (list of dict per position) → DataFrame
    with rows = positions, columns = amino acids.
    """
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


# ---------------------------------------------------------
# 3. Plot Expansion / Compaction logos
# ---------------------------------------------------------

def plot_expansion_compaction_logos(
    csv_path,
    k,
    min_count=20,
    use_abs_weight=False,
    clip_z=None,
):
    """
    Recreate Expansion/Compaction Logos from window_omissions_all.csv
    for motifs of length k.
    """
    mean_effect = computeWeightedResidueEffects_from_csv(
        csv_path,
        k=k,
        min_count=min_count,
        use_abs_weight=use_abs_weight,
        clip_z=clip_z,
    )

    df = effects_to_df(mean_effect)

    # Positive entries → expansion; negative → compaction
    df_expand  = df.clip(lower=0.0)      # keep positive, zero out negative
    df_compact = (-df).clip(lower=0.0)   # flip sign and keep positive

    # Expansion Logo
    plt.figure(figsize=(12, 4))
    logomaker.Logo(df_expand, color_scheme='chemistry')
    plt.xlabel("Position in Motif")
    plt.ylabel("Enrichment (scaled)")
    plt.title(f"Expansion Logo")
    plt.tight_layout()
    plt.show()

    # Compaction Logo
    plt.figure(figsize=(12, 4))
    logomaker.Logo(df_compact, color_scheme='chemistry')
    plt.xlabel("Position in Motif")
    plt.ylabel("Enrichment (scaled)")
    plt.title(f"Compaction Logo")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4. Example call
# ---------------------------------------------------------

if __name__ == "__main__":
    csv_path = "interpretingUnirep/window_omissions_all.csv"  # adjust path if needed
    k = 10                                  # motif length you want
    plot_expansion_compaction_logos(
        csv_path,
        k=k,
        min_count=50,      # raise/lower depending on dataset size
        use_abs_weight=False,
        clip_z=3.0         # optional: clip extreme z-scores
    )
