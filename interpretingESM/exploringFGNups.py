import pickle
import matplotlib.pyplot as plt
import numpy as np
from allOcclusionScripts import sliding_embedding_occlusion_with_fragments, get_layer3_embeddings
import joblib


regrModel = joblib.load('krr.joblib')
pca = joblib.load('esm_pca2.joblib')
with open("allEffects4.pkl", "rb") as f:
    allEffects = pickle.load(f)
k = 5
effects = allEffects[k - 1]
color_map = {
    # Charged
    'R': 'gray', 'K': 'gray', 'D': 'gray', 'E': 'gray',

    # Polar uncharged
    'S': 'gray', 'T': 'gray', 'N': 'gray', 'Q': 'gray',

    # Hydrophobic
    'A': 'gray', 'V': 'gray', 'L': 'gray', 'I': 'gray',
    'M': 'gray', 'F': 'dodgerblue', 'Y': 'gray', 'W': 'gray',

    # Special
    'G': 'forestgreen', 'P': 'gray', 'C': 'gray'
}
nup98idx = 12

print(len(effects[13]))

nup98 = effects[nup98idx]
seq = 'GCFNKSFGTPFGGGTGGFGTTSTFGQNTGFGTTSGGAFGTSAFGSSNNTGGLFGNSQTKPGGLFGTSSFSQPATSTSTGFGFGTSTGTANTLFGTASTGTSLFSSQNNAFAQNKPTGFGNFGTSTSSGGLFGTTNTTSNPFGSTSGSLFGPUA'
emb = get_layer3_embeddings(seq)
k = 3
allEffects, allFrgs = sliding_embedding_occlusion_with_fragments(seq, emb, k, pca, regrModel, 'zero')
plt.figure(figsize=(12,4))
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title(f"Per-residue impact on predicted Rg, Nup98, k = {k}")
plt.show()

seq = 'GCQTSRGLFGNNNTNNINNSSSGMNNASAGLFGSKPUA'
emb = get_layer3_embeddings(seq)
allEffects, allFrgs = sliding_embedding_occlusion_with_fragments(seq, emb, k, pca, regrModel, 'zero')
plt.figure(figsize=(12,4))
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title(f"Per-residue impact on predicted Rg, Nup49, k = {k}")
plt.show()

seq = 'GCPSASPAFGANQTPTFGQSQGASQPNPPGFGSISSSTALFPTGSQPAPPTFGTVSSSSQPPVFGQQPSQSAFGSGTTPNUA'
emb = get_layer3_embeddings(seq)
allEffects, allFrgs = sliding_embedding_occlusion_with_fragments(seq, emb, k, pca, regrModel, 'zero')
plt.figure(figsize=(12,4))
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title(f"Per-residue impact on predicted Rg, Nup153 NUS Domain, k = {k}")
plt.show()

seq = 'GCGFKGFDTSSSSSNSAASSSFKFGVSSSSSGPSQTLTSTGNFKFGDQGGFKIGVSSDSGSINPMSEGFKFSKPIGDFKFGVSSESKPEEVKKDSKNDNFKFGLSSGLSNPVUA'
emb = get_layer3_embeddings(seq)
allEffects, allFrgs = sliding_embedding_occlusion_with_fragments(seq, emb, k, pca, regrModel, 'zero')
plt.figure(figsize=(12,4))
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.bar(range(len(seq) - k + 1), allEffects, color=[color_map.get(aa, 'white') for aa in seq])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title(f"Per-residue impact on predicted Rg, Nup153 NUL Domain, k = {k}")
plt.show()
