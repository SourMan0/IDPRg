from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import csv
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

def partial_dependence_1d(model, X, feature_index, grid_points=50):
    X = np.asarray(X)
    x_vals = np.linspace(X[:, feature_index].min(), 
                         X[:, feature_index].max(), 
                         grid_points)

    pd_vals = np.zeros(grid_points)

    for i, val in enumerate(x_vals):
        X_temp = X.copy()
        X_temp[:, feature_index] = val
        pd_vals[i] = model.predict(X_temp).mean()

    return x_vals, pd_vals
def pd_directionality(pd_vals, x_vals):
    grads = np.gradient(pd_vals, x_vals)
    return grads.mean() 

feature_names = [
    "Fraction hydrophobic", "Fraction Polar", "Fraction Aromatic", "Fraction Proline", "Fraction Glycine",  
    "Fraction Charged", "Log Length", "Kappa", 
]
feature_names_abridge = [
    'FracHydroPhob', 'FracPol', 'FracArom', 'FracProl', 'FracGly', 'FracCharged', "LogLen", "Hydropathy:fcr",  "kappa", ]

seed = 46
ridge_alphas = np.logspace(-3, 3, 10)
lasso_alphas = np.logspace(-4, 1, 10)

krr_alphas = [0.001, 0.01, 0.1, 1, 10]
krr_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
outlierIndices =  [123, 136, 151, 158, 171, 185]
inl = np.ones(190, dtype=bool)
inl[outlierIndices] = False
y = []
with open('../training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            y.append(row[4])
        counter += 1
X2 = []
with open('../physicalFeatures/protein_features3St.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            X2.append(row[1:])
        counter += 1        
X2 = np.asarray(X2, dtype=np.float64)
X2 = X2[inl]

lin2 = LinearRegression().fit(X2, y)

best_ridge2 = GridSearchCV(Ridge(), {"alpha": ridge_alphas}, cv=5, scoring="r2")
best_ridge2.fit(X2, y)

best_lasso = GridSearchCV(Lasso(max_iter=10000), {"alpha": lasso_alphas}, cv=5, scoring="r2")
best_lasso.fit(X2, y)

best_krr2 = GridSearchCV(
KernelRidge(kernel="rbf"),
{"alpha": krr_alphas, "gamma": krr_gammas},
            cv=5, scoring="r2", n_jobs=-1
        )
best_krr2.fit(X2, y)

lin_coef2 = lin2.coef_
ridge_coef2 = best_ridge2.best_estimator_.coef_
lasso_coef = best_lasso.best_estimator_.coef_

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)

n_features = X2.shape[1]
direction = np.zeros(n_features)

for j in range(n_features):
    x_vals, pd_vals = partial_dependence_1d(best_krr2.best_estimator_, X2, j)
    direction[j] = pd_directionality(pd_vals, x_vals)

n_features2 = X2.shape[1]
direction2 = np.zeros(n_features2)

for j in range(n_features2):
    x_vals, pd_vals = partial_dependence_1d(gpr, X2, j)
    direction2[j] = pd_directionality(pd_vals, x_vals)
print("Standardized inputs")
print("\n=== Linear Regression Coefficients ===")
for i in np.argsort(-np.abs(lin_coef2)):
    print(f"{feature_names[i]:25s}  coef={lin_coef2[i]:+7.3f}")

print("\n=== Ridge Regression Coefficients ===")
for i in np.argsort(-np.abs(ridge_coef2)):
    print(f"{feature_names[i]:25s}  coef={ridge_coef2[i]:+7.3f}")





idx = np.argsort(ridge_coef2)        # ascending
features_sorted = np.array(feature_names)[idx]
coefs_sorted = np.array(ridge_coef2)[idx]

plt.figure(figsize=(8, 5))
plt.bar(features_sorted, coefs_sorted)
plt.axhline(0, color='black', linewidth=1)
plt.title("Ridge Regression Feature Coefficients")
plt.ylabel("Coefficient")
plt.xticks(rotation=45, ha = "right")
plt.tight_layout()
plt.show()

idx = np.argsort(direction)
features_sorted = np.array(feature_names)[idx]
direction = np.array(direction)[idx]

plt.figure(figsize=(10,5))
plt.bar(feature_names, direction)
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1)
plt.ylabel("Directionality (mean gradient of partial dependence)")
plt.title("KRR Feature Directionality")
plt.tight_layout()
plt.show()

idx = np.argsort(direction2)
features_sorted = np.array(feature_names)[idx]
direction = np.array(direction2)[idx]

plt.figure(figsize=(10,5))
plt.bar(feature_names, direction)
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1)
plt.ylabel("Directionality (mean gradient of partial dependence)")
plt.title("KRR Feature Directionality")
plt.tight_layout()
plt.show()



idx = np.argsort(lin_coef2)        # ascending
features_sorted = np.array(feature_names)[idx]
coefs_sorted = np.array(lin_coef2)[idx]

plt.figure(figsize=(8, 5))
plt.bar(features_sorted, coefs_sorted)
plt.axhline(0, color='black', linewidth=1)
plt.title("Linear RegressionFeature Coefficients")
plt.ylabel("Coefficient")
plt.xticks(rotation=45, ha = "right")
plt.tight_layout()
plt.show()

idx = np.argsort(lasso_coef)        # ascending
features_sorted = np.array(feature_names)[idx]
coefs_sorted = np.array(lasso_coef)[idx]

plt.figure(figsize=(8, 5))
plt.bar(features_sorted, coefs_sorted)
plt.axhline(0, color='black', linewidth=1)
plt.title("Lasso Feature Coefficients")
plt.ylabel("Coefficient")
plt.xticks(rotation=45, ha = "right")
plt.tight_layout()
plt.show()



