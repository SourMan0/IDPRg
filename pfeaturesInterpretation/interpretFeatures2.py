# LEAK-FREE rewrite of interpretFeatures2.py
# ------------------------------------------
# Previously this script consumed protein_features3St2.csv, which had already
# been StandardScaler.fit_transform'd across all 167 sequences in
# extractFeaturesFromPoints3St.py before any train/test split happened. The
# GridSearchCV α selection then ran over pre-scaled features, so the picked
# alpha (and the ridge coefficients shown in the figure) reflected an
# all-data scaling.
#
# Now the CSV stores raw features. We wrap StandardScaler+model in a sklearn
# Pipeline that's passed to GridSearchCV(cv=5); the scaler is refit per inner
# CV fold so α is selected without leakage. The final pipeline is then refit
# on all 163 inliers (refit=True by default) -- that's the model whose
# coefficients are interpreted. The coefficient magnitudes are still in
# z-units (because StandardScaler still sits in front of the linear model),
# but now the z was computed on training data only at each step of the CV.

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    "Fraction hydrophobic", "Fraction Polar", "Fraction Aromatic",
    "Fraction Proline", "Fraction Glycine", "Fraction Charged",
    "Log Length", "Kappa",
]

seed = 46
ridge_alphas = np.logspace(-3, 3, 10)
lasso_alphas = np.logspace(-4, 1, 10)
krr_alphas = [0.001, 0.01, 0.1, 1, 10]
krr_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]

outlierIndices = [114, 125, 137, 163]
inl = np.ones(167, dtype=bool)
inl[outlierIndices] = False

# Target: column 4 of inliers.csv == "Rg normalized w/0.406 (nm)"
# (matches the unchanged behaviour of the previous version of this script)
y = []
with open('../training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        y.append(row[4])
y = np.asarray(y, dtype=np.float64)

# Raw features (no longer pre-scaled in the CSV)
X2 = []
with open('../physicalFeatures/protein_features3St2.csv', newline='') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        X2.append(row[1:])
X2 = np.asarray(X2, dtype=np.float64)
X2 = X2[inl]
print(f'X2 shape (inliers, raw): {X2.shape}')


def with_scaler(model):
    """Pipeline that scales then fits the model (leak-free in CV)."""
    return Pipeline([('scaler', StandardScaler()), ('model', model)])


# ------------------------------------------------------------------
# Fits — α / γ selected via leak-free Pipeline-based GridSearchCV
# ------------------------------------------------------------------
lin2 = with_scaler(LinearRegression()).fit(X2, y)

best_ridge2 = GridSearchCV(with_scaler(Ridge()),
                           {"model__alpha": ridge_alphas},
                           cv=5, scoring="r2").fit(X2, y)

best_lasso = GridSearchCV(with_scaler(Lasso(max_iter=10000)),
                          {"model__alpha": lasso_alphas},
                          cv=5, scoring="r2").fit(X2, y)

best_krr2 = GridSearchCV(with_scaler(KernelRidge(kernel="rbf")),
                         {"model__alpha": krr_alphas,
                          "model__gamma": krr_gammas},
                         cv=5, scoring="r2", n_jobs=-1).fit(X2, y)

print(f'Best Ridge α: {best_ridge2.best_params_["model__alpha"]:.4g}  '
      f'(CV R² = {best_ridge2.best_score_:.4f})')
print(f'Best Lasso α: {best_lasso.best_params_["model__alpha"]:.4g}  '
      f'(CV R² = {best_lasso.best_score_:.4f})')
print(f'Best KRR α={best_krr2.best_params_["model__alpha"]}, '
      f'γ={best_krr2.best_params_["model__gamma"]}  '
      f'(CV R² = {best_krr2.best_score_:.4f})')

# Coefficients live on the "model" step inside the Pipeline. Because
# StandardScaler sits in front of the linear model, the coefs are still in
# z-units (per-feature std) — comparable to the previous figure's units.
lin_coef2 = lin2.named_steps['model'].coef_
ridge_coef2 = best_ridge2.best_estimator_.named_steps['model'].coef_
lasso_coef = best_lasso.best_estimator_.named_steps['model'].coef_

# KRR has no .coef_ -- use partial-dependence directionality instead.
# Wrap a fresh GPR in the same scaler so its inputs match the fitted scale.
n_features = X2.shape[1]
direction = np.zeros(n_features)
for j in range(n_features):
    x_vals, pd_vals = partial_dependence_1d(best_krr2.best_estimator_, X2, j)
    direction[j] = pd_directionality(pd_vals, x_vals)

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gpr = with_scaler(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                            alpha=1e-6, random_state=42)).fit(X2, y)
direction2 = np.zeros(n_features)
for j in range(n_features):
    x_vals, pd_vals = partial_dependence_1d(gpr, X2, j)
    direction2[j] = pd_directionality(pd_vals, x_vals)

print("Standardised inputs (Pipeline scaler; leak-free)")

print("\n=== Linear Regression Coefficients ===")
for i in np.argsort(-np.abs(lin_coef2)):
    print(f"{feature_names[i]:25s}  coef={lin_coef2[i]:+7.3f}")

print("\n=== Ridge Regression Coefficients ===")
for i in np.argsort(-np.abs(ridge_coef2)):
    print(f"{feature_names[i]:25s}  coef={ridge_coef2[i]:+7.3f}")

print("\n=== Lasso Regression Coefficients ===")
for i in np.argsort(-np.abs(lasso_coef)):
    print(f"{feature_names[i]:25s}  coef={lasso_coef[i]:+7.3f}")


# ------------------------------------------------------------------
# Journal figure — ridge coefficients, sorted, in-text size
# ------------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
})

feature_names_arr = np.array(feature_names)
coefs = np.array(ridge_coef2)
idx = np.argsort(coefs)
features_sorted = feature_names_arr[idx]
coefs_sorted = coefs[idx]

pos_color = "#6F7F99"
neg_color = "#A9B2BF"
colors, alphas = [], []
for c in coefs_sorted:
    if c >= 0:
        colors.append(pos_color)
        alphas.append(0.72)
    else:
        colors.append(neg_color)
        alphas.append(0.85)

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for f, c, col, al in zip(features_sorted, coefs_sorted, colors, alphas):
    ax.bar(f, c, width=0.6, color=col, alpha=al, edgecolor="none")
ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
ax.set_ylabel("Ridge coefficient")
ax.set_title("Reference distribution of ridge coefficients")
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", rotation=45, length=2)
ax.tick_params(axis="y", length=2)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
plt.tight_layout()
plt.savefig("reference_ridge_coefficients_sorted.pdf", bbox_inches="tight")
plt.savefig("reference_ridge_coefficients_sorted.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print('-> reference_ridge_coefficients_sorted.pdf (+ .png)')


# ------------------------------------------------------------------
# Supplementary plots (KRR / GPR directionality, Linear / Lasso bars)
# ------------------------------------------------------------------
def _save_bar(values, title, path):
    idx = np.argsort(values)
    sorted_names = feature_names_arr[idx]
    sorted_vals = np.array(values)[idx]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sorted_names, sorted_vals)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(title)
    ax.set_ylabel('value')
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path[:-4] + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'-> {path} (+ .png)')


_save_bar(direction, "KRR feature directionality (∂PD / ∂x mean)",
          "krr_feature_directionality.pdf")
_save_bar(direction2, "GPR feature directionality (∂PD / ∂x mean)",
          "gpr_feature_directionality.pdf")
_save_bar(lin_coef2, "Linear regression feature coefficients",
          "linear_feature_coefficients.pdf")
_save_bar(lasso_coef, "Lasso feature coefficients",
          "lasso_feature_coefficients.pdf")
