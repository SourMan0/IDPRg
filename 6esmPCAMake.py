import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


x = torch.load('esm_embeddings/esm6layer.pt')
print(x.shape)

X_np = np.array(x.detach().cpu())


target_dims = [190, 100, 50, 20, 10]
n_layers = X_np.shape[1]

pca_results = {}

for layer in range(n_layers):
    print(f"\nLayer {layer}:")
    X_layer = X_np[:, layer, :]  # (190, 320)
    X_scaled = StandardScaler().fit_transform(X_layer)

    pca_results[layer] = {}
    for d in target_dims:
        if d > X_layer.shape[1]:
            continue  # skip if n_components > features
        pca = PCA(n_components=d, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_results[layer][d] = X_pca
        print(f"  PCA {d} → {X_pca.shape}")

# Flatten all PCA outputs horizontally (same row order)

for layer, dims in pca_results.items():
    for dim, arr in dims.items():
        filename = f"6esmPCA/layer{layer}_pca{dim}.npy"
        np.save(filename, arr.astype(np.float32))
        print(f"Saved {filename} with shape {arr.shape}")