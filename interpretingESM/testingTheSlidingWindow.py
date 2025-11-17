import joblib
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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


# The mask token "X" is something ESM recognizes as "unknown"
def sliding_mask_effect(seq, model, tokenizer, pca, gpr, layer_idx=4, window=5,  mask_token="X", step = 1 ):
    """
    Returns ΔRg per position:
      +Δ → region promotes expansion
      -Δ → region promotes compaction
    """

    # Helper: extract mean-pooled embedding for a given sequence
    def get_layer_embedding(seq, layer_idx):
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        emb = out.hidden_states[layer_idx].mean(dim=1).squeeze()
        return emb.cpu().numpy()

    # --- Baseline prediction ---
    baseline_emb = get_layer_embedding(seq, layer_idx)
    baseline_pca = pca.transform(baseline_emb.reshape(1, -1))
    baseline_pred = gpr.predict(baseline_pca)[0]

    # --- Sliding mask loop ---
    effects = np.zeros(len(seq))
    masked_positions = []
    masked_fragments = []

    # --- Sliding window loop ---
    for start in range(0, len(seq) - window + 1, step):
        mask_slice = list(range(start, start + window))
        masked_fragment = seq[start:start + window]
        masked_positions.append(mask_slice)
        masked_fragments.append(masked_fragment)

        # Create masked sequence
        masked_seq = list(seq)
        for j in mask_slice:
            masked_seq[j] = mask_token
        masked_seq = "".join(masked_seq)
        # Compute masked prediction
        masked_emb = get_layer_embedding(masked_seq, layer_idx)
        masked_pca = pca.transform(masked_emb.reshape(1, -1))
        masked_pred = gpr.predict(masked_pca)[0]

        # Centered ΔRg attribution
        center_idx = start + window // 2
        delta = baseline_pred - masked_pred
        effects[center_idx] = delta

    return effects, masked_positions, masked_fragments

'''
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
'''