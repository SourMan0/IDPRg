# Best Model: ['2' 'Rg norm w/0.5' 'pH regr out' 'All' '2' 'Kernel Ridge' '90/10' '0.2383644152070975' '0.04300319174291359']
# Even though it says that the second model was the best, which excludes three features, we'll go with the first one for robustness
# We're also going to intepret the column where nothing is regressed out to avoid negative Rg values. The best model for this was linear, but I'll also try ridge and kernel ridge

# This is our next best model with these considerations:
# ['5' 'Rg norm w/0.5' 'No regr out' 'Inliers' '1' 'Linear' '90/10', '0.30180289682393613' '0.04837609987441006']

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import csv
from sklearn.inspection import permutation_importance

feature_names = [
    "Fraction hydrophobic", "Fraction Polar", "Fraction Aromatic", "Fraction Proline", "Fraction Glycine",  "Fraction Non-charged",
    "Fraction Charged", "Charge Asymmetry", "Mean Hydropathy", "Variance of Hydropathy", "MolWt", "Log Length", "Hydropathy:fcr", "fraction aromatic: fraction hydrophobic", "Kappa", "SCD", "SHD"
]
seed = 46
ridge_alphas = np.logspace(-3, 3, 10)
krr_alphas = [0.001, 0.01, 0.1, 1, 10]
krr_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
outlierIndices =  [123, 136, 151, 158, 171, 185]
inl = np.ones(190, dtype=bool)
inl[outlierIndices] = False
y = []
with open('C:\\Users\\saleh\\Documents\\Python\\NPC-GNN\\IDPregression\\IDPRg\\training\\inliers.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            y.append(row[4])
        counter += 1

X = []
with open('../physicalFeatures/protein_features.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            X.append(row[1:])
        counter += 1        
X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.float64)
X = X[inl]
print(np.shape(X))
X2 = []
with open('../physicalFeatures/protein_featuresSt.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            X2.append(row[1:])
        counter += 1        
X2 = np.asarray(X2, dtype=np.float64)
X2 = X2[inl]

lin = LinearRegression().fit(X, y)

best_ridge = GridSearchCV(Ridge(), {"alpha": ridge_alphas}, cv=5, scoring="r2")
best_ridge.fit(X, y)

best_krr = GridSearchCV(
KernelRidge(kernel="rbf"),
{"alpha": krr_alphas, "gamma": krr_gammas},
            cv=5, scoring="r2", n_jobs=-1
        )
best_krr.fit(X, y)

lin_coef = lin.coef_
ridge_coef = best_ridge.best_estimator_.coef_
r = permutation_importance(best_krr.best_estimator_, X, y, n_repeats=20, random_state=0)
perm_importance = r.importances_mean

print("\n=== Linear Regression Coefficients ===")
for i in np.argsort(-np.abs(lin_coef)):
    print(f"{feature_names[i]:25s}  coef={lin_coef[i]:+7.3f}")

print("\n=== Ridge Regression Coefficients ===")
for i in np.argsort(-np.abs(ridge_coef)):
    print(f"{feature_names[i]:25s}  coef={ridge_coef[i]:+7.3f}")

print("\n=== Kernel Ridge Permutation Importance ===")
for i in np.argsort(-perm_importance):
    print(f"{feature_names[i]:25s}  importance={perm_importance[i]:.4f}")


lin2 = LinearRegression().fit(X2, y)

best_ridge2 = GridSearchCV(Ridge(), {"alpha": ridge_alphas}, cv=5, scoring="r2")
best_ridge2.fit(X2, y)

best_krr2 = GridSearchCV(
KernelRidge(kernel="rbf"),
{"alpha": krr_alphas, "gamma": krr_gammas},
            cv=5, scoring="r2", n_jobs=-1
        )
best_krr2.fit(X2, y)

lin_coef2 = lin2.coef_
ridge_coef2 = best_ridge2.best_estimator_.coef_
r2 = permutation_importance(best_krr2.best_estimator_, X2, y, n_repeats=20, random_state=0)
perm_importance2 = r2.importances_mean

print("Standardized inputs")
print("\n=== Linear Regression Coefficients ===")
for i in np.argsort(-np.abs(lin_coef2)):
    print(f"{feature_names[i]:25s}  coef={lin_coef[i]:+7.3f}")

print("\n=== Ridge Regression Coefficients ===")
for i in np.argsort(-np.abs(ridge_coef2)):
    print(f"{feature_names[i]:25s}  coef={ridge_coef[i]:+7.3f}")

print("\n=== Kernel Ridge Permutation Importance ===")
for i in np.argsort(-perm_importance2):
    print(f"{feature_names[i]:25s}  importance={perm_importance[i]:.4f}")
