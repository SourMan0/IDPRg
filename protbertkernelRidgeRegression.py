import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def krr(inpath, test_size=0.1, target_col="Rg (nm)",
        alphas=[1e-2, 1e-1, 1, 10],
        gammas=[1e-3, 1e-2, 1e-1, 1],
        outpath=None):

    df = pd.read_csv(inpath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {inpath}")

    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' in CSV. Found: {list(df.columns)}")

    # numeric PCA columns as features
    numeric_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c != target_col]
    print(f"Using {len(numeric_cols)} numeric columns for Kernel Ridge Regression.")

    X = df[numeric_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf"))
    ])

    param_grid = {
        "krr__alpha": alphas,
        "krr__gamma": gammas
    }

    model = GridSearchCV(pipe, param_grid, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    test_rmse = ((y_test - y_pred_test) ** 2).mean() ** 0.5
    train_rmse = ((y_train - y_pred_train) ** 2).mean() ** 0.5
    alpha = model.best_params_['krr__alpha']
    gamma = model.best_params_['krr__gamma']

    print(f"Test RMSE: {test_rmse}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Chosen alpha: {alpha}")
    print(f"Chosen gamma: {gamma}")

    Path("results").mkdir(exist_ok=True)
    if outpath is None:
        outpath = f"results/{Path(inpath).stem}_KRRResults.txt"

    with open(outpath, "w") as f:
        f.write("Kernel Ridge Regression Results\n")
        f.write(f"Chosen alpha: {alpha}\n")
        f.write(f"Chosen gamma: {gamma}\n")
        f.write(f"Test RMSE: {test_rmse}\n")
        f.write(f"Train RMSE: {train_rmse}\n")

    print(f"Saved results → {outpath}")

if __name__ == "__main__":
    inpath = "data/protbertEmbeddingsPCA10_withTarget.csv"
    test_size = 0.1
    target_col = "Rg (nm)"
    alphas = [1e-3, 1e-2, 1e-1, 1, 10]
    gammas = [1e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    krr(inpath, test_size, target_col, alphas, gammas)