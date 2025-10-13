from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
from pathlib import Path

def linear_regression(inpath, test_size, emb_col, alphas=[0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 10000], outpath=None):
    # df = pd.read_csv(inpath)
    df = pd.read_csv(inpath, converters={emb_col: ast.literal_eval})

    X = df[emb_col].tolist()
    y = df["Rg (nm)"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])

    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    test_rmse = ((y_test - y_pred_test) ** 2).mean() ** 0.5
    train_rmse = ((y_train - y_pred_train) ** 2).mean() ** 0.5
    alpha = pipe.named_steps['ridge'].alpha_
    params = pipe.named_steps['ridge'].get_params()

    print(f"Test RMSE: {test_rmse}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Chosen alpha: {alpha}")

    if outpath is None:
        outpath = f"results/{Path(inpath).stem}TestSize{test_size}LinearResults.txt"

    with open(outpath, "w") as f:
        f.write("Linear Regression Results\n")
        f.write(f"Test RMSE: {test_rmse}\n")
        f.write(f"Train RMSE: {train_rmse}\n")


if __name__ == "__main__":
    # Settings
    inpath = "data/unirepEmbeddings.csv"
    test_size = 0.2
    emb_col = "UniRep Embedding"
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2500, 3250, 5000, 7500, 10000]

    linear_regression(inpath, test_size, emb_col, alphas)