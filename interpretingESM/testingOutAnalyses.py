import joblib
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from testingTheSlidingWindow import sliding_mask_effect


color_map = {
    # Charged
    'R': 'dodgerblue', 'K': 'blue', 'D': 'red', 'E': 'orangered',

    # Polar uncharged
    'S': 'lightgreen', 'T': 'limegreen', 'N': 'green', 'Q': 'forestgreen',

    # Hydrophobic
    'A': 'gray', 'V': 'dimgray', 'L': 'darkgray', 'I': 'slategray',
    'M': 'silver', 'F': 'black', 'Y': 'brown', 'W': 'maroon',

    # Special
    'G': 'gold', 'P': 'orange', 'C': 'yellow'
}

model = joblib.load('esm_gpr.joblib')
pca = joblib.load('esm_pca.joblib')

model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name)

sample_seq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'

effects, masked_positions, masked_fragmets = sliding_mask_effect(sample_seq, esm_model, tokenizer, pca, model)
effects2, masked_positions2, masked_fragmets2 = sliding_mask_effect(sample_seq, esm_model, tokenizer, pca, model, window=4)


plt.figure(figsize=(12,4))
plt.bar(range(len(sample_seq)), effects, color=[color_map.get(aa, 'white') for aa in sample_seq])
plt.bar(range(len(sample_seq)), effects, color=[color_map.get(aa, 'white') for aa in sample_seq])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("Per-residue impact on predicted Rg")
plt.show()

plt.figure(figsize=(12,4))
plt.bar(range(len(sample_seq)), effects2, color=[color_map.get(aa, 'white') for aa in sample_seq])
plt.bar(range(len(sample_seq)), effects2, color=[color_map.get(aa, 'white') for aa in sample_seq])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("Per-residue impact on predicted Rg")
plt.show()

r, _ = pearsonr(effects2, effects)

print(r)


motifs, inverse = np.unique(masked_fragmets, return_inverse=True)
n_motifs = len(motifs)

print(f"Motifs Length: {len(motifs)}, inverse length: {len(inverse)}, effects length: {len(effects)}, fragments length: {len(masked_fragmets)}")
effects = effects[2:-2]

means = np.zeros(n_motifs)
stds = np.zeros(n_motifs)
counts = np.zeros(n_motifs, dtype=int)


for i, m in enumerate(motifs):
    vals = effects[inverse == i]
    means[i] = vals.mean()
    stds[i] = vals.std(ddof=1) if len(vals) > 1 else 0.0
    counts[i] = len(vals)

z_scores = np.zeros_like(means)
for i in range(n_motifs):
    if counts[i] > 1 and stds[i] > 0:
        z_scores[i] = means[i] / (stds[i] / np.sqrt(counts[i]))
    else:
        z_scores[i] = means[i] / 1

sorted_idx = np.argsort(means)[::-1]  # descending
motifs_sorted = motifs[sorted_idx]
means_sorted = means[sorted_idx]
stds_sorted = stds[sorted_idx]
counts_sorted = counts[sorted_idx]
z_sorted = z_scores[sorted_idx]

print(f"{'Motif':<6} {'Mean ΔRg':>10} {'Count':>8} {'Z-score':>10}")
for i in range(len(motifs_sorted)):
    print(f"{motifs_sorted[i]:<6} {means_sorted[i]:>10.4f} {counts_sorted[i]:>8} {z_sorted[i]:>10.2f}")


# Check that all fragments are same length
k = len(masked_fragmets[0])
assert all(len(f) == k for f in masked_fragmets), "All fragments must be same length!"

# Split into Rg-increasing vs decreasing sets
pos_mask = effects > 0
neg_mask = effects < 0

masked_fragmets = np.array(masked_fragmets)
effects = np.array(effects)

pos_frags, pos_weights = masked_fragmets[pos_mask], effects[pos_mask]
neg_frags, neg_weights = masked_fragmets[neg_mask], np.abs(effects[neg_mask])

# -------------------------------
# 2️⃣  Build weighted profiles
# -------------------------------
aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
aa_to_idx = {a: i for i, a in enumerate(aa)}

def weighted_profile(seqs, weights):
    """Build position × amino-acid weighted frequency matrix"""
    k = len(seqs[0])
    prof = np.zeros((k, 20))
    for s, w in zip(seqs, weights):
        for i, ch in enumerate(s):
            prof[i, aa_to_idx[ch]] += w
    # Normalize within each position
    prof /= np.sum(prof, axis=1, keepdims=True) + 1e-9
    return prof

pos_prof = weighted_profile(pos_frags, pos_weights)
neg_prof = weighted_profile(neg_frags, neg_weights)

# -------------------------------
# 3️⃣  Compute positional enrichment
# -------------------------------
enrichment = np.log2((pos_prof + 1e-6) / (neg_prof + 1e-6))

# -------------------------------
# 4️⃣  Visualize as heatmap
# -------------------------------
plt.figure(figsize=(10, 10))
im = plt.imshow(enrichment.T, cmap="coolwarm", aspect="auto", origin="lower",
                vmin=-2, vmax=2)

plt.colorbar(im, label="log2(Rg↑ / Rg↓ enrichment)")
plt.xticks(np.arange(k), [f"Pos{i+1}" for i in range(k)])
plt.yticks(np.arange(20), aa)
plt.xlabel("Position in motif")
plt.ylabel("Amino acid")
plt.title("Positional enrichment of residues for Rg-increasing vs Rg-decreasing fragments")
plt.tight_layout()
plt.show()


'''
def diagnose_mask_effects(fragments, effects):
    print("=== ΔRg Diagnostics ===")
    print(f"Total fragments: {len(fragments)}")
    pos_mask = effects > 0
    neg_mask = effects < 0
    n_pos, n_neg = np.sum(pos_mask), np.sum(neg_mask)

    print(f"Positive (Rg↑): {n_pos} fragments")
    print(f"Negative (Rg↓): {n_neg} fragments")

    if n_pos == 0:
        print("⚠️ No positive ΔRg values found. The heatmap will be meaningless — check your sign convention.")
        return
    if n_neg == 0:
        print("⚠️ No negative ΔRg values found. You’re only seeing one class — verify ΔRg definition.")
        return

    # ------------------------
    # 1️⃣  Distribution check
    # ------------------------
    plt.figure(figsize=(5,3))
    plt.hist(effects, bins=50, color='gray')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Distribution of ΔRg values")
    plt.xlabel("ΔRg")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ------------------------
    # 2️⃣  Summary statistics
    # ------------------------
    mean_pos, mean_neg = np.mean(effects[pos_mask]), np.mean(effects[neg_mask])
    std_pos, std_neg = np.std(effects[pos_mask]), np.std(effects[neg_mask])
    print(f"Mean ΔRg (positive set): {mean_pos:+.4f} ± {std_pos:.4f}")
    print(f"Mean ΔRg (negative set): {mean_neg:+.4f} ± {std_neg:.4f}")

    imbalance_ratio = n_pos / (n_neg + 1e-9)
    if imbalance_ratio < 0.25:
        print("⚠️ Strong imbalance: very few positive fragments relative to negative ones. "
              "This can cause artificial enrichment (blue bias).")
    elif imbalance_ratio > 4:
        print("⚠️ Strong imbalance: very few negative fragments relative to positive ones.")

    # ------------------------
    # 3️⃣  Sign inversion check
    # ------------------------
    print("\nChecking possible sign inversion:")
    example_idx = np.argmax(np.abs(effects))
    print(f"Example fragment: {fragments[example_idx]}, ΔRg = {effects[example_idx]:+.4f}")
    print("If this fragment’s masking should *lower* Rg but ΔRg is positive, your sign is flipped.")

    # ------------------------
    # 4️⃣  Normalization bias hint
    # ------------------------
    print("\nNormalization advice:")
    print(" - If most fragments have ΔRg < 0, normalize profiles globally (sum over all positions).")
    print(" - You can also downsample negatives to match positives before building the heatmap.\n")

    print("=== End diagnostics ===")

# Run the diagnostic
diagnose_mask_effects(masked_fragmets, effects)

'''


aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
aa_to_idx = {a: i for i, a in enumerate(aa)}

def weighted_profile(seqs, weights):
    """Position × amino-acid weighted frequency matrix (global normalization)"""
    k = len(seqs[0])
    prof = np.zeros((k, 20))
    for s, w in zip(seqs, weights):
        for i, ch in enumerate(s):
            prof[i, aa_to_idx[ch]] += w
    prof /= np.sum(prof) + 1e-9  # global normalization
    return prof


def balanced_profiles(fragments, effects, seed=42):
    """Balance positive/negative sets and compute normalized profiles"""
    np.random.seed(seed)

    # separate by sign
    pos_mask = effects > 0
    neg_mask = effects < 0
    pos_frags, pos_weights = fragments[pos_mask], effects[pos_mask]
    neg_frags, neg_weights = fragments[neg_mask], np.abs(effects[neg_mask])

    # balance sample sizes
    n = min(len(pos_frags), len(neg_frags))
    if n == 0:
        raise ValueError("Not enough positive or negative samples to balance.")
    idx_pos = np.random.choice(len(pos_frags), n, replace=False)
    idx_neg = np.random.choice(len(neg_frags), n, replace=False)

    pos_frags, pos_weights = pos_frags[idx_pos], pos_weights[idx_pos]
    neg_frags, neg_weights = neg_frags[idx_neg], neg_weights[idx_neg]

    # normalize weights to equal total contribution
    pos_weights /= np.sum(pos_weights) + 1e-9
    neg_weights /= np.sum(neg_weights) + 1e-9

    pos_prof = weighted_profile(pos_frags, pos_weights)
    neg_prof = weighted_profile(neg_frags, neg_weights)

    return pos_prof, neg_prof


def plot_enrichment(pos_prof, neg_prof, title="Positional enrichment for Rg↑ vs Rg↓"):
    enrichment = np.log2((pos_prof + 1e-6) / (neg_prof + 1e-6))
    k = enrichment.shape[0]

    plt.figure(figsize=(10, 10))
    im = plt.imshow(enrichment.T, cmap="coolwarm", aspect="auto", origin="lower",
                    vmin=-2, vmax=2)
    plt.colorbar(im, label="log2(Rg↑ / Rg↓ enrichment)")
    plt.xticks(np.arange(k), [f"Pos{i+1}" for i in range(k)])
    plt.yticks(np.arange(20), aa)
    plt.xlabel("Position in motif")
    plt.ylabel("Amino acid")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- Run the full process ---
pos_prof, neg_prof = balanced_profiles(masked_fragmets, effects)
plot_enrichment(pos_prof, neg_prof)