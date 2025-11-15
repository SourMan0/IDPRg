# Best Model: ['2' 'Rg norm w/0.418' 'No regr out' 'All' 'ESM-12' '4' '190' 'GPR' '90/10' '0.5539548399198044' '0.03291141668342302']

import csv
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import joblib
from sklearn.decomposition import PCA
import torch


y = []
# Choosing all the points with no regr out just to aboid any negative numbers. 

with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            y.append(row[4])
        counter += 1

y = np.array(y, dtype=float)
# Need to choose layer four and PCA 190. 
x = torch.load('../esm_embeddings/esm12layer.pt')
X_np = np.array(x.detach().cpu())
X_4 = X_np[:, 4, :]
pca = PCA(n_components=190, random_state=42)
X = pca.fit_transform(X_4)
joblib.dump(pca, "esm_pca.joblib")


kernel = (C(1.0, (1e-3, 1e3)) *RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))+ WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1)))
model = GaussianProcessRegressor(kernel=kernel,alpha=1e-6,n_restarts_optimizer=5,normalize_y=True,random_state=42)
model.fit(X, y)
joblib.dump(model, "esm_gpr.joblib")