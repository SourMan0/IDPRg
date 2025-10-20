from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast
from pathlib import Path

def lasso_regression(inpath, test_size, emb_col, rg_col="Rg (nm)", alphas=[0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 10000], outpath=None):
    # Special handling for UniRep (embedding column is a stringified list)
    df = pd.read_csv(inpath, converters={emb_col: ast.literal_eval})

    # X: list-of-lists -> numpy array; y -> numpy array (so math works reliably)
    X = np.asarray(df[emb_col].tolist(), dtype=float)
    y = df[rg_col].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(alphas=alphas, cv=5, max_iter=100000, n_jobs=-1, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    alpha = pipe.named_steps['lasso'].alpha_
    ss_res_test = ((y_test - y_pred_test) ** 2).sum()
    ss_tot_test = ((y_test - y_test.mean()) ** 2).sum()
    test_r2 = 1 - ss_res_test / ss_tot_test if ss_tot_test != 0 else float("nan")

    ss_res_train = ((y_train - y_pred_train) ** 2).sum()
    ss_tot_train = ((y_train - y_train.mean()) ** 2).sum()
    train_r2 = 1 - ss_res_train / ss_tot_train if ss_tot_train != 0 else float("nan")

    print(f"Test RMSE: {test_rmse}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test R^2: {test_r2}")
    print(f"Train R^2: {train_r2}")
    print(f"Chosen alpha: {alpha}")
    
    if outpath is None:
        outpath = f"results/{Path(inpath).stem}TestSize{test_size}LassoResults.txt"

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write("LASSO Regression Results\n")
        f.write(f"Chosen alpha: {alpha}\n")
        f.write(f"Test RMSE: {test_rmse}\n")
        f.write(f"Train RMSE: {train_rmse}\n")
        f.write(f"Test R^2: {test_r2}\n")
        f.write(f"Train R^2: {train_r2}\n")


if __name__ == "__main__":
    # Settings
    inpath = "data/unirep_allNormalized.csv"
    test_size = 0.2
    emb_col = "UniRep Embedding"
    rg_col = "Normalized Rg with ν = 0.427 (nm)"
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2500, 3250, 5000, 7500, 10000]

    lasso_regression(inpath, test_size, emb_col, rg_col, alphas)
