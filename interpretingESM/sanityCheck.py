import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt

# Load ESM
tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model.eval()

def get_embeddings(seq, layer_idx=4):
    """Return per-residue embeddings for a sequence at a given layer."""
    tokens = tok(seq, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    # hidden_states is [layer][batch, seq_len, 320]
    emb = out.hidden_states[layer_idx][0]  # shape (L+2, 320)
    return emb[1:-1]  # strip BOS/EOS

def cosine_sim(A, B):
    """Compute cosine similarity residue-by-residue."""
    A = A / A.norm(dim=-1, keepdim=True)
    B = B / B.norm(dim=-1, keepdim=True)
    return (A * B).sum(-1).cpu().numpy()

# Example protein sequence
seq = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"

# Step 1: original embeddings
orig = get_embeddings(seq, layer_idx=4)

# Step 2: X-masked sequence
start, window = 10, 5
seq_masked = seq[:start] + "X" * window + seq[start+window:]

# Step 3: masked embeddings
masked = get_embeddings(seq_masked, layer_idx=4)

# Step 4: compute cosine similarity per residue
sim = cosine_sim(orig, masked)

print("Per-residue cosine similarity:")
print(sim)
print("Mean similarity:", sim.mean())

plt.plot(sim)
plt.show()