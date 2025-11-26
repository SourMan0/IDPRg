import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
import matplotlib.pyplot as plt
import csv
import pickle

###############################################
# 1. LOAD ESM MODEL (layer 4 is index = 4)
###############################################

tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model.eval()

regrModel = joblib.load('esm_gpr.joblib')
pca = joblib.load('esm_pca.joblib')

###############################################
# 2. GET LAYER-4 EMBEDDINGS FOR A SEQUENCE
###############################################

def get_layer4_embeddings(seq):
    """
    Returns per-residue embeddings from layer 4 (shape L x 320).
    Strips BOS/EOS automatically.
    """
    tokens = tok(seq, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    # hidden_states: list of length 13 (0–12), each shape (1, L+2, 320)
    emb_l4 = out.hidden_states[4][0]  # choose layer 4
    emb_l4 = emb_l4[1:-1]            # strip BOS/EOS → (L, 320)
    return emb_l4


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

def run_occlusion_all_sequences(sequences, embeddings, k_values, pca, reg_model, method="mean"):
    """
    sequences: list of strings
    embeddings: list of tensors (each (L, D) )
    k_values: list of window sizes

    returns:
        all_dRg[k][i]  → ΔRg array for sequence i
        all_frags[k][i] → list of fragments for sequence i
    """
    all_dRg = [[] for k in k_values]
    all_frags = [[] for k in k_values]

    for i, (seq, E) in enumerate(zip(sequences, embeddings)):
        for k in k_values:
            drg, frags = sliding_embedding_occlusion_with_fragments(
                seq, E, k, pca, reg_model, method
            )
            all_dRg[k - 1].append(drg)
            all_frags[k - 1].append(frags)

    return all_dRg, all_frags

sequences = []
with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
        counter += 1

embeddings = []
for i in sequences:
    embeddings.append(get_layer4_embeddings(i))
allEffects, allFragments = run_occlusion_all_sequences(sequences, embeddings, list(range(1,11)), pca, regrModel, method = 'zero')

with open("allEffects3.pkl", "wb") as f:
    pickle.dump(allEffects, f)
with open("allFragments3.pkl", "wb") as f:
    pickle.dump(allFragments, f)

