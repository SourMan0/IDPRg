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

# Final headline configs: two targets, both with no covariate regressed out.
#   - "0.5"  -> column 3  ("Rg normalized w/0.5 (nm)")  -- KRR R²=0.221, RMSE=0.060
#   - "0.406"-> column 4  ("Rg normalized w/0.406 (nm)") -- KRR R²=0.192, RMSE=0.097
# Pick which target this run produces via the FIG_TARGET environment variable
# (defaults to 0.5). Output filenames are suffixed _target0p5 / _target0p406.
import os
TARGET = os.environ.get("FIG_TARGET", "0.5")
TARGET_COL = {"0.5": 3, "0.406": 4}[TARGET]
OUT_DIR = "target_0p5" if TARGET == "0.5" else "target_0p406"
os.makedirs(OUT_DIR, exist_ok=True)
def _out(name):
    return os.path.join(OUT_DIR, name)
print(f"Target = Rg normalized w/{TARGET}, no regr out (col {TARGET_COL}); "
      f"figures -> {OUT_DIR}/")

y = []
with open('../training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        y.append(row[TARGET_COL])
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

def _journal_style_bar(values, ylabel, title, path):
    """Bar plot styled to match the journal Ridge-coefficient figure.

    Same compact figsize (3.4 × 2.4), same muted positive / light-grey
    negative colour palette, same sort order (most negative → most positive),
    same axis cleanup. Used for both the Ridge headline figure and the KRR
    feature-directionality plot so they're visually comparable.
    """
    idx = np.argsort(values)
    sorted_names = feature_names_arr[idx]
    sorted_vals = np.array(values)[idx]

    colors, alphas = [], []
    for v in sorted_vals:
        if v >= 0:
            colors.append("#6F7F99")
            alphas.append(0.72)
        else:
            colors.append("#A9B2BF")
            alphas.append(0.85)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for name, val, col, al in zip(sorted_names, sorted_vals, colors, alphas):
        ax.bar(name, val, width=0.6, color=col, alpha=al, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=45, length=2)
    ax.tick_params(axis="y", length=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path[:-4] + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"-> {path} (+ .png)")


# Headline Ridge figure (unchanged content, refactored to share the helper)
_journal_style_bar(ridge_coef2, "Ridge coefficient",
                   "Reference distribution of ridge coefficients",
                   _out("reference_ridge_coefficients_sorted.pdf"))

# KRR directionality figure, styled identically to the Ridge one
_journal_style_bar(direction, "KRR directionality",
                   "Reference distribution of KRR directionality",
                   _out("reference_krr_directionality.pdf"))


# ------------------------------------------------------------------
# SHAP-based KRR interpretation (replaces the single-scalar directionality
# above for the journal-quality KRR panel).
#
# Two figures:
#   1. Global importance: mean(|SHAP|) per feature -- styled like the Ridge
#      headline figure, with bars coloured by the sign of the mean signed
#      SHAP (so direction and magnitude are decoupled rather than averaged
#      together).
#   2. Beeswarm: per-instance SHAP attribution coloured by raw feature
#      value -- shows whether the effect is monotonic, U-shaped, or
#      heterogeneous across the 163 inliers.
#
# We use KernelExplainer because KRR has no built-in SHAP support. The
# background distribution is summarised to 25 k-means medoids so that the
# explainer's reference expectation is well-conditioned without being too
# slow.
# ------------------------------------------------------------------
import shap

print('\n>>> Computing SHAP values for the KRR pipeline...')
background = shap.kmeans(X2, 25)
explainer = shap.KernelExplainer(best_krr2.best_estimator_.predict, background)
# Explain every inlier. silent=True suppresses the per-sample progress bar.
shap_vals = explainer.shap_values(X2, silent=True, nsamples='auto')
shap_vals = np.asarray(shap_vals)                # (163, 8)
mean_abs_shap = np.abs(shap_vals).mean(axis=0)   # global importance

# Direction: sign of corr(feature_raw, SHAP). If raising the feature value
# tends to raise its SHAP (= push the prediction up), correlation is
# positive and we colour the bar like a "positive ridge coefficient".
# Otherwise we colour it like a negative one.  This is the same signal the
# beeswarm encodes via dot colour, just collapsed to a single sign per
# feature for the journal bar plot.
shap_direction_sign = np.zeros(shap_vals.shape[1])
for j in range(shap_vals.shape[1]):
    feat_col = X2[:, j]
    if feat_col.std() < 1e-12:
        shap_direction_sign[j] = 0.0
    else:
        shap_direction_sign[j] = np.corrcoef(feat_col, shap_vals[:, j])[0, 1]

print('  Global importance (mean |SHAP|) and direction (sign corr(feat, SHAP)):')
for i in np.argsort(-mean_abs_shap):
    print(f'    {feature_names[i]:25s}  mean|SHAP|={mean_abs_shap[i]:.4f}  '
          f'direction corr={shap_direction_sign[i]:+.3f}')


def _shap_signed_bar(mean_abs, direction, path):
    """Signed SHAP importance, styled to match the Ridge journal figure.

    Bar height = mean(|SHAP|) * sign(corr(feature, SHAP)). Negative bars
    point below zero (features whose increase compacts the chain),
    positive bars point above (features whose increase expands it).
    Bars sorted from most negative (left) to most positive (right) so the
    layout reads like the Ridge figure.
    """
    signed = mean_abs * np.sign(direction)
    idx = np.argsort(signed)
    sorted_names = feature_names_arr[idx]
    sorted_vals = signed[idx]

    colors, alphas = [], []
    for v in sorted_vals:
        if v >= 0:
            colors.append("#6F7F99")
            alphas.append(0.72)
        else:
            colors.append("#A9B2BF")
            alphas.append(0.85)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for name, val, col, al in zip(sorted_names, sorted_vals, colors, alphas):
        ax.bar(name, val, width=0.6, color=col, alpha=al, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("signed mean |SHAP|")
    ax.set_title("Reference distribution of KRR SHAP importance")
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=45, length=2)
    ax.tick_params(axis="y", length=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path[:-4] + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"-> {path} (+ .png)")


_shap_signed_bar(mean_abs_shap, shap_direction_sign,
                 _out("reference_krr_shap_importance.pdf"))


# Supplementary beeswarm
fig = plt.figure(figsize=(6, 4))
shap.summary_plot(shap_vals, X2, feature_names=feature_names,
                  plot_type="dot", show=False, color_bar=True)
plt.tight_layout()
plt.savefig(_out("krr_shap_beeswarm.pdf"), bbox_inches="tight")
plt.savefig(_out("krr_shap_beeswarm.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
print(f'-> {_out("krr_shap_beeswarm.pdf")} (+ .png)')


# ------------------------------------------------------------------
# Supplementary plots (GPR directionality, Linear / Lasso bars) — kept
# at the larger exploratory size since they're not headline figures.
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


# KRR directionality already has a journal-styled version above; only keep
# the GPR exploratory plot here since GPR has no headline counterpart.
_save_bar(direction2, "GPR feature directionality (∂PD / ∂x mean)",
          _out("gpr_feature_directionality.pdf"))

# Linear and Lasso coefficient figures, formatted in the same journal style
# as the Ridge headline and KRR-SHAP plots.
_journal_style_bar(lin_coef2, "Linear coefficient",
                   "Reference distribution of linear coefficients",
                   _out("reference_linear_coefficients_sorted.pdf"))
_journal_style_bar(lasso_coef, "Lasso coefficient",
                   "Reference distribution of lasso coefficients",
                   _out("reference_lasso_coefficients_sorted.pdf"))
