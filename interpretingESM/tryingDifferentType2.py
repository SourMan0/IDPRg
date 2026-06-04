import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
import matplotlib.pyplot as plt
import csv
import pickle

###############################################
# 1. LOAD ESM MODEL (layer 3 of esm2_t6, which is index = 3)
###############################################

tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
model.eval()

# Single artifact now: Pipeline([StandardScaler, PCA, KernelRidge]) fit
# without leakage by prepareForInterpretation.py. pipeline.predict applies
# scaler -> pca -> krr in one shot, so the helper below no longer needs to
# load (and align) a separate PCA object.
pipeline = joblib.load('krr_pipeline.joblib')

###############################################
# 2. GET LAYER-1 EMBEDDINGS FOR A SEQUENCE
###############################################
# Layer chosen to match the leak-free best config from the new sweep
# (ESM-6 layer 1, PCA=100). The pipeline joblib expects 320-d features
# from this exact layer.
ESM_LAYER = 1


def get_layer_embeddings(seq):
    """Per-residue embeddings from ESM_LAYER, shape (L, 320), BOS/EOS stripped."""
    tokens = tok(seq, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    emb = out.hidden_states[ESM_LAYER][0]
    emb = emb[1:-1]
    return emb


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

def predict_rg_from_embedding(E, pipe):
    """E: (L, D); pipe: fitted sklearn Pipeline(StandardScaler, PCA, model).

    Returns: predicted Rg (float)
    """
    pooled = E.mean(dim=0).cpu().numpy()
    pred = pipe.predict(pooled.reshape(1, -1))[0]
    return float(pred)


def sliding_embedding_occlusion_with_fragments(seq, E, k, pipe, method="mean"):
    """Slide a window of size k along the embedding, occlude, and record ΔRg."""
    L = len(seq)
    drg_list = []
    fragments = []

    rg_orig = predict_rg_from_embedding(E, pipe)

    for start in range(L - k + 1):
        E_occ = occlude_embedding_window(E, start, k, method)
        rg_occ = predict_rg_from_embedding(E_occ, pipe)
        drg_list.append(rg_occ - rg_orig)
        fragments.append(seq[start:start + k])

    return np.array(drg_list), fragments


def run_occlusion_all_sequences(sequences, embeddings, k_values, pipe, method="mean"):
    all_dRg = [[] for _ in k_values]
    all_frags = [[] for _ in k_values]
    for i, (seq, E) in enumerate(zip(sequences, embeddings)):
        for k in k_values:
            drg, frags = sliding_embedding_occlusion_with_fragments(seq, E, k, pipe, method)
            all_dRg[k - 1].append(drg)
            all_frags[k - 1].append(frags)
    return all_dRg, all_frags


sequences = []
with open('../training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
        counter += 1

embeddings = []
for i in sequences:
    embeddings.append(get_layer_embeddings(i))
allEffects, allFragments = run_occlusion_all_sequences(
    sequences, embeddings, list(range(1, 11)), pipeline, method='zero'
)

with open("allEffects4.pkl", "wb") as f:
    pickle.dump(allEffects, f)
with open("allFragments4.pkl", "wb") as f:
    pickle.dump(allFragments, f)
