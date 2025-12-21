import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ast
from pathlib import Path

def pca_calc(inpath, pca_num_components, emb_col="UniRep Embedding"):
    #df = pd.read_csv(inpath)    # Standard CSV read

    # Special handling for Unirep (embedding is a list as a string)
    df = pd.read_csv(inpath, converters={emb_col: ast.literal_eval})

    emb = df[emb_col].tolist()
    pca_num_components = min(pca_num_components, len(emb))

    print(f"Reducing dimensions from {len(emb[0])} to {pca_num_components}")

    # Define PCA and scaler
    pca = PCA(n_components=pca_num_components, svd_solver="randomized", random_state=42)
    scaler = StandardScaler(with_mean=True)

    # Transform embeddings
    emb = scaler.fit_transform(emb)
    emb = pca.fit_transform(emb)

    df[emb_col] = emb.tolist()
    df.to_csv(f"data/{Path(inpath).stem}PCA{pca_num_components}.csv", index=False)

if __name__ == "__main__":
    # Settings
    inpath = "data/unirep_allRaw.csv"
    pca_num_components = [10, 20, 50, 100, 167]
    emb_col = "UniRep Embedding"

    for n_components in pca_num_components:
        pca_calc(inpath, n_components, emb_col)

    inpath = "data/unirep_inliersRaw.csv"
    pca_num_components = [10, 20, 50, 100, 163]
    emb_col = "UniRep Embedding"

    for n_components in pca_num_components:
        pca_calc(inpath, n_components, emb_col)