import pickle
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

regr_model = joblib.load('esm_gpr.joblib')
pca = joblib.load('esm_pca.joblib')
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

tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model.eval()

def occlude_embedding_window(E, start, k, method="mean"):
    """
    E: (L, D) embedding matrix
    start: window start index
    k: window size
    method: "mean", "zero", or "noise"
    """
    L, D = E.shape
    E_occ = E.clone()

    if method == "mean":
        filler = E.mean(dim=0, keepdim=True)   # (1, D)
        E_occ[start:start+k] = filler

    elif method == "zero":
        E_occ[start:start+k] = 0.0

    elif method == "noise":
        mu = E.mean(dim=0)
        sigma = E.std(dim=0)
        noise = torch.normal(mu, sigma, size=(k, D))
        E_occ[start:start+k] = noise

    return E_occ

def predict_rg_from_embedding(E, pca, reg_model):
    """
    E: (L, D)
    pca: fitted PCA object
    reg_model: your regression model (kernel ridge, GPR, etc.)

    Returns: predicted Rg (float)
    """
    # Mean pool → 1D vector
    pooled = E.mean(dim=0).cpu().numpy()

    # Apply PCA transform
    X = pca.transform(pooled.reshape(1, -1))

    # Predict Rg
    pred = reg_model.predict(X)[0]
    return float(pred)

def sliding_embedding_occlusion_with_fragments(seq, E, k, pca, reg_model, method="mean"):
    """
    seq: string protein sequence
    E: (L, D) tensor of embeddings for this sequence
    k: window size
    pca, reg_model: fitted models
    method: occlusion style

    returns:
        drg_list: list of ΔRg values (length L-k+1)
        fragments: list of sequence fragments (length L-k+1)
    """
    L = len(seq)
    drg_list = []
    fragments = []

    # reference prediction
    rg_orig = predict_rg_from_embedding(E, pca, reg_model)

    for start in range(L - k + 1):
        E_occ = occlude_embedding_window(E, start, k, method)
        rg_occ = predict_rg_from_embedding(E_occ, pca, reg_model)

        drg = rg_occ - rg_orig
        drg_list.append(drg)

        frag = seq[start:start+k]
        fragments.append(frag)

    return np.array(drg_list), fragments

nup153nus = 'GCPSASPAFGANQTPTFGQSQGASQPNPPGFGSISSSTALFPTGSQPAPPTFGTVSSSSQPPVFGQQPSQSAFGSGTTPNUA'
nup153nul = 'GCGFKGFDTSSSSSNSAASSSFKFGVSSSSSGPSQTLTSTGNFKFGDQGGFKIGVSSDSGSINPMSEGFKFSKPIGDFKFGVSSESKPEEVKKDSKNDNFKFGLSSGLSNPVUA'
nup98 = 'GCFNKSFGTPFGGGTGGFGTTSTFGQNTGFGTTSGGAFGTSAFGSSNNTGGLFGNSQTKPGGLFGTSSFSQPATSTSTGFGFGTSTGTANTLFGTASTGTSLFSSQNNAFAQNKPTGFGNFGTSTSSGGLFGTTNTTSNPFGSTSGSLFGPUA'
nup49 = 'GCQTSRGLFGNNNTNNINNSSSGMNNASAGLFGSKPUA'

tokens = tok(nup153nus, return_tensors="pt", add_special_tokens=True)
with torch.no_grad():
    out = model(**tokens, output_hidden_states=True)
# hidden_states: list of length 13 (0–12), each shape (1, L+2, 320)
emb_l4 = out.hidden_states[4][0]  # choose layer 4
emb_l4 = emb_l4[1:-1]   

effects, frags = sliding_embedding_occlusion_with_fragments(nup153nus, emb_l4, 5, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup153nus) - 4), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.bar(range(len(nup153nus) - 4), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("Per-residue impact on predicted Rg")
plt.show()

effects, frags = sliding_embedding_occlusion_with_fragments(nup153nus, emb_l4, 1, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup153nus)), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.bar(range(len(nup153nus)), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("Per-residue impact on predicted Rg, k = 1")
plt.show()

effects, frags = sliding_embedding_occlusion_with_fragments(nup153nus, emb_l4, 2, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup153nus) - 1), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.bar(range(len(nup153nus) - 1), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("Per-residue impact on predicted Rg, k = 2")
plt.show()

effects, frags = sliding_embedding_occlusion_with_fragments(nup153nus, emb_l4, 3, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup153nus) - 2), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.bar(range(len(nup153nus) - 2), effects, color=[color_map.get(aa, 'white') for aa in nup153nus])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("Per-residue impact on predicted Rg, k = 3")
plt.show()

tokens = tok(nup98, return_tensors="pt", add_special_tokens=True)
with torch.no_grad():
    out = model(**tokens, output_hidden_states=True)
# hidden_states: list of length 13 (0–12), each shape (1, L+2, 320)
emb_l4 = out.hidden_states[4][0]  # choose layer 4
emb_l4 = emb_l4[1:-1]   

effects, frags = sliding_embedding_occlusion_with_fragments(nup98, emb_l4, 5, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup98) - 4), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.bar(range(len(nup98) - 4), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("nup98 Per-residue impact on predicted Rg, k = 5")
plt.show()

effects, frags = sliding_embedding_occlusion_with_fragments(nup98, emb_l4, 1, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup98)), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.bar(range(len(nup98)), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("nup98 Per-residue impact on predicted Rg, k = 1")
plt.show()

effects, frags = sliding_embedding_occlusion_with_fragments(nup98, emb_l4, 2, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup98) - 1), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.bar(range(len(nup98) - 1), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("nup98 Per-residue impact on predicted Rg, k = 2")
plt.show()

effects, frags = sliding_embedding_occlusion_with_fragments(nup98, emb_l4, 3, pca, regr_model, method="zero")
plt.figure(figsize=(12,4))
plt.bar(range(len(nup98) - 2), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.bar(range(len(nup98) - 2), effects, color=[color_map.get(aa, 'white') for aa in nup98])
plt.xlabel("Residue index")
plt.ylabel("ΔRg (Å)")
plt.title("nup98 Per-residue impact on predicted Rg, k = 3")
plt.show()