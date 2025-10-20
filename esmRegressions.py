import numpy as np
import csv

features = {}
layers6 = range(7)
PCAvals = [190, 100, 50, 20, 10]

for l in layers6:
    features[l] = {}
    for p in PCAvals:
        features[l][p] = np.load(f'6esmPCA/layer{l}_pca{p}.npy', allow_pickle=True)

features12 = {}
layers12 = range(13)
for l in layers12:
    features12[l] = {}
    for p in PCAvals:
        features12[l][p] = np.load(f'12esmPCA/layer{l}_pca{p}.npy', allow_pickle=True)