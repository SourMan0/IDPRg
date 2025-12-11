# Best Model: ['2' 'Rg norm w/0.418' 'No regr out' 'All' 'ESM-12' '4' '190' 'GPR' '90/10' '0.5539548399198044' '0.03291141668342302']

import csv
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import joblib
from sklearn.decomposition import PCA
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV

outlierIndices =  [114, 125, 137, 163]
inl = np.ones(167, dtype=bool)
inl[outlierIndices] = False
y = []
# Choosing all the points with no regr out just to avoid any negative numbers. 


#Try two models: ESM-12 layer 6 w/PCA 50 and Lasso
# or ESM-6 layer 4 w/PCA 167 and Kernel Ridge Regression
# Both were on the inliers w/0.5 reg
with open('../training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            y.append(row[3])
        counter += 1

y = np.array(y, dtype=float)
# Need to choose layer four and PCA 190. 
x = torch.load('../esmScripts/esm_embeddings/esm12layer.pt')
X_np = np.array(x.detach().cpu())
X_6 = X_np[:, 6, :]
X_6 = X_6[inl]
pca = PCA(n_components=50, random_state=42)
X = pca.fit_transform(X_6)
joblib.dump(pca, "esm_pca.joblib")
lasso_alphas = np.logspace(-4, 1, 10)

best_lasso = GridSearchCV(Lasso(max_iter=10000), {"alpha": lasso_alphas}, cv=5, scoring="r2")
best_lasso.fit(X, y)
joblib.dump(best_lasso, 'lasso.joblib')

#I'll try another one here:
#This time it will be PCA = 100, inliers ESM-7, layer 3, kernel ridge regression
x = torch.load('../esmScripts/esm_embeddings/esm6layer.pt')
X_np = np.array(x.detach().cpu())
X_3 = X_np[:, 3, :]
X_3 = X_3[inl]
pca = PCA(n_components=100, random_state=42)
X = pca.fit_transform(X_3)
joblib.dump(pca, "esm_pca2.joblib")
krr_alphas = [0.001, 0.01, 0.1, 1, 10]
krr_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]


best_krr = GridSearchCV(
            KernelRidge(kernel="rbf"),
            {"alpha": krr_alphas, "gamma": krr_gammas},
            cv=5, scoring="r2", n_jobs=-1
        )
best_krr.fit(X, y)

joblib.dump(best_krr, "krr.joblib")
'''
kernel = (C(1.0, (1e-3, 1e3)) *RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))+ WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1)))
model = GaussianProcessRegressor(kernel=kernel,alpha=1e-6,n_restarts_optimizer=5,normalize_y=True,random_state=42)
model.fit(X, y)
joblib.dump(model, "esm_gpr.joblib")
'''