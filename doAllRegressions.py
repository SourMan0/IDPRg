#Move to parent directory to regenerate scripts
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_models_rmse(X, y, random_state=42):
    splits = {
        "80/20": 0.2,
        "85/15": 0.15,
        "90/10": 0.10
    }
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ridge_alphas = np.logspace(-3, 3, 10)
    lasso_alphas = np.logspace(-4, 1, 10)
    krr_alphas = [0.001, 0.01, 0.1, 1, 10]
    krr_gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]

    results = []

    for split_name, test_size in splits.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 1️⃣ Linear Regression
        lin = LinearRegression().fit(X_train, y_train)
        y_pred = lin.predict(X_test)
        results.append(["Linear", split_name, r2_score(y_test, y_pred),
                        np.sqrt(mean_squared_error(y_test, y_pred))])

        # 2️⃣ Ridge Regression
        best_ridge = GridSearchCV(Ridge(), {"alpha": ridge_alphas}, cv=5, scoring="r2")
        best_ridge.fit(X_train, y_train)
        y_pred = best_ridge.predict(X_test)
        results.append(["Ridge", split_name, r2_score(y_test, y_pred),
                        np.sqrt(mean_squared_error(y_test, y_pred))])

        # 3️⃣ Lasso Regression
        best_lasso = GridSearchCV(Lasso(max_iter=10000), {"alpha": lasso_alphas}, cv=5, scoring="r2")
        best_lasso.fit(X_train, y_train)
        y_pred = best_lasso.predict(X_test)
        results.append(["Lasso", split_name, r2_score(y_test, y_pred),
                        np.sqrt(mean_squared_error(y_test, y_pred))])

        # 4️⃣ Kernel Ridge Regression
        best_krr = GridSearchCV(
            KernelRidge(kernel="rbf"),
            {"alpha": krr_alphas, "gamma": krr_gammas},
            cv=5, scoring="r2", n_jobs=-1
        )
        best_krr.fit(X_train, y_train)
        y_pred = best_krr.predict(X_test)
        results.append(["Kernel Ridge", split_name, r2_score(y_test, y_pred),
                        np.sqrt(mean_squared_error(y_test, y_pred))])

        # 5️⃣ Gaussian Process Regression
        if np.shape(X) == (190, 190):
            kernel = (C(1.0, (1e-3, 1e3)) *RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))+ WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1)))
            gpr = GaussianProcessRegressor(kernel=kernel,alpha=1e-6,n_restarts_optimizer=5,normalize_y=True,random_state=42)
        else:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gpr.fit(X_train, y_train)
        y_pred = gpr.predict(X_test)
        results.append(["GPR", split_name, r2_score(y_test, y_pred),
                        np.sqrt(mean_squared_error(y_test, y_pred))])

    # Print clean summary
    '''
    print(f"\n{'Model':15s} {'Split':8s} {'R²':>6s} {'RMSE':>8s}")
    print("-" * 40)
    for m, s, r2, rmse in results:
        print(f"{m:15s} {s:8s} {r2:6.3f} {rmse:8.3f}")
    '''

    return results