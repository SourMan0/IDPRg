import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
import matplotlib.pyplot as plt
import csv
###############################################
# 1. LOAD ESM MODEL (layer 4 is index = 4)
###############################################

tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model.eval()

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


###############################################
# 3. OCCLUSION FUNCTION (ZERO or MEAN)
###############################################

def occlude(emb, start, window, mode="zero"):
    """
    emb: (L, 320) tensor
    start: window start index (0-based)
    window: window size
    mode: 'zero' or 'mean'
    """
    emb2 = emb.clone()

    if mode == "zero":
        emb2[start:start+window] = 0.0

    elif mode == "mean":
        mean_vec = emb.mean(dim=0, keepdim=True)
        emb2[start:start+window] = mean_vec

    else:
        raise ValueError("mode must be 'zero' or 'mean'")

    return emb2


###############################################
# 4. CONVERT EMBEDDINGS → FEATURES FOR REGRESSOR
###############################################

def embeddings_to_features(emb, pca=None):
    """
    emb: (L, 320)
    Produces (1, D) feature vector for your regressor.
    """
    pooled = emb.mean(dim=0).unsqueeze(0).numpy()  # shape (1, 320)

    if pca is not None:
        pooled = pca.transform(pooled)

    return pooled


###############################################
# 5. PREDICT Rg FROM FEATURES
###############################################

def predict_rg(emb, regressor, pca=None):
    """
    emb: (L, 320) tensor
    regressor: your sklearn model (Ridge, KRR, GPR, etc.)
    """
    X = embeddings_to_features(emb, pca)  # numpy array
    return regressor.predict(X)[0]


###############################################
# 6. SLIDING OCCLUSION → ΔRg ARRAY
###############################################

def sliding_occlusion_rg(seq, regressor, pca=None, window=5, mode="zero"):
    """
    Returns ΔRg for each window start position.
    """
    emb = get_layer4_embeddings(seq)               # shape (L, 320)
    baseline = predict_rg(emb, regressor, pca)

    L = len(seq)
    deltas = []
    fragments = []
    indices = []

    # 2. Slide the window
    for start in range(L - window + 1):
        end = start + window

        # Record the fragment sequence
        frag = seq[start:end]
        fragments.append(frag)
        indices.append((start, end))

        # Occlude in embedding space
        emb_occ = occlude(emb, start, window, mode)
        rg_occ = predict_rg(emb_occ, regressor, pca)

        deltas.append(rg_occ - baseline)

    return np.array(deltas), baseline, fragments, indices

sequences = []
with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
            print(" " in row[0], counter)
        counter += 1
for seq in sequences[:5]:
    regr_model = joblib.load('esm_gpr.joblib')
    pca = joblib.load('esm_pca.joblib')

    effects, baseleine, fragments, indices = sliding_occlusion_rg(seq, regr_model, pca)

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

    plt.figure(figsize=(12,4))
    plt.bar(range(len(seq) - 4), effects, color=[color_map.get(aa, 'white') for aa in seq])
    plt.bar(range(len(seq) - 4), effects, color=[color_map.get(aa, 'white') for aa in seq])
    plt.xlabel("Residue index")
    plt.ylabel("ΔRg (Å)")
    plt.title("Per-residue impact on predicted Rg")
    plt.show()

