#Move to parent directory to regenerate scripts
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def _max_pca_dim(n_components, n_features, n_train, cv=5):
    cv_floor = (n_train * (cv - 1)) // cv
    return max(1, min(n_components, n_features, cv_floor))


def _build_pipeline(model, n_components=None, do_scaling=False):
    """Pipeline that does StandardScaler -> PCA -> model, fit only after split."""
    steps = []
    if do_scaling or n_components is not None:
        steps.append(("scaler", StandardScaler()))
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_models_rmse(X, y, random_state=42, n_components=None, do_scaling=False):
    """Train/test split first, then fit every preprocessor + model on train only.

    Modes:
    - n_components=None, do_scaling=False (default): legacy behaviour, fits the
      raw model on X_train directly (used by callers that already pre-process).
    - n_components=None, do_scaling=True: Pipeline(StandardScaler, model).
      For physical features: leak-free scaler fit per train fold.
    - n_components=<int>: Pipeline(StandardScaler, PCA, model) regardless of
      do_scaling. Used by the ESM sweep.
    """
    splits = {
        "80/20": 0.2,
        "85/15": 0.15,
        "90/10": 0.10,
    }
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ridge_alphas = np.logspace(-3, 3, 10)
    lasso_alphas = np.logspace(-4, 1, 10)
    krr_alphas = [0.001, 0.01, 0.1, 1, 10]
    krr_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]

    use_pipeline = (n_components is not None) or do_scaling
    results = []

    for split_name, test_size in splits.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if n_components is not None:
            d_used = _max_pca_dim(n_components, X_train.shape[1], X_train.shape[0])
        else:
            d_used = None

        def mk(model):
            return _build_pipeline(model, n_components=d_used, do_scaling=do_scaling)

        # 1. Linear Regression
        if use_pipeline:
            est = mk(LinearRegression())
            est.fit(X_train, y_train)
        else:
            est = LinearRegression().fit(X_train, y_train)
        y_pred = est.predict(X_test)
        row = ["Linear", split_name, r2_score(y_test, y_pred),
               np.sqrt(mean_squared_error(y_test, y_pred))]
        if n_components is not None: row.append(d_used)
        results.append(row)

        # 2. Ridge
        if use_pipeline:
            gs = GridSearchCV(mk(Ridge()), {"model__alpha": ridge_alphas},
                              cv=5, scoring="r2")
        else:
            gs = GridSearchCV(Ridge(), {"alpha": ridge_alphas}, cv=5, scoring="r2")
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        row = ["Ridge", split_name, r2_score(y_test, y_pred),
               np.sqrt(mean_squared_error(y_test, y_pred))]
        if n_components is not None: row.append(d_used)
        results.append(row)

        # 3. Lasso
        if use_pipeline:
            gs = GridSearchCV(mk(Lasso(max_iter=10000)),
                              {"model__alpha": lasso_alphas},
                              cv=5, scoring="r2")
        else:
            gs = GridSearchCV(Lasso(max_iter=10000), {"alpha": lasso_alphas},
                              cv=5, scoring="r2")
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        row = ["Lasso", split_name, r2_score(y_test, y_pred),
               np.sqrt(mean_squared_error(y_test, y_pred))]
        if n_components is not None: row.append(d_used)
        results.append(row)

        # 4. Kernel Ridge
        # In pipeline mode we leave n_jobs=1 so the caller can parallelise the
        # outer loop without oversubscribing CPUs.
        if use_pipeline:
            gs = GridSearchCV(mk(KernelRidge(kernel="rbf")),
                              {"model__alpha": krr_alphas,
                               "model__gamma": krr_gammas},
                              cv=5, scoring="r2", n_jobs=1)
        else:
            gs = GridSearchCV(KernelRidge(kernel="rbf"),
                              {"alpha": krr_alphas, "gamma": krr_gammas},
                              cv=5, scoring="r2", n_jobs=-1)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        row = ["Kernel Ridge", split_name, r2_score(y_test, y_pred),
               np.sqrt(mean_squared_error(y_test, y_pred))]
        if n_components is not None: row.append(d_used)
        results.append(row)

        # 5. Gaussian Process Regression
        if (n_components is not None and d_used >= 100) or \
           (not use_pipeline and np.shape(X) == (190, 190)):
            kernel = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                      + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1)))
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                           n_restarts_optimizer=5, normalize_y=True,
                                           random_state=42)
        else:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                           alpha=1e-6, random_state=42)
        if use_pipeline:
            est = mk(gpr)
            est.fit(X_train, y_train)
        else:
            est = gpr
            est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        row = ["GPR", split_name, r2_score(y_test, y_pred),
               np.sqrt(mean_squared_error(y_test, y_pred))]
        if n_components is not None: row.append(d_used)
        results.append(row)

    return results
