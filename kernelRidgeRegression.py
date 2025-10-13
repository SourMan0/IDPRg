from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
from pathlib import Path

def krr(inpath, test_size, emb_col, alphas=[1e-2, 1e-1, 1, 10], gammas=[1e-3, 1e-2, 1e-1, 1], outpath=None):
    # df = pd.read_csv(inpath)  # Standard CSV read
    
    # Special handling for Unirep (embedding is a list as a string)
    df = pd.read_csv(inpath, converters={emb_col: ast.literal_eval})

    X = df[emb_col].tolist()
    y = df["Rg (nm)"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf"))
    ])

    param_grid = {
    "krr__alpha": alphas,
    "krr__gamma": gammas  # RBF width (1/(2σ²))
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

    if outpath is None:
        outpath = f"results/{Path(inpath).stem}TestSize{test_size}KernelRidgeResults.txt"

    with open(outpath, "w") as f:
        f.write("Kernel Ridge Regression Results\n")
        f.write(f"Chosen alpha: {alpha}\n")
        f.write(f"Chosen gamma: {gamma}\n")
        f.write(f"Test RMSE: {test_rmse}\n")
        f.write(f"Train RMSE: {train_rmse}\n")


if __name__ == "__main__":
    # Settings
    inpath = "data/unirepEmbeddingsPCA10.csv"
    test_size = 0.1
    emb_col = "UniRep Embedding"
    alphas = [1e-3, 1e-2, 1e-1, 1, 10]
    gammas = [1e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    krr(inpath, test_size, emb_col, alphas, gammas)