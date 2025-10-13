import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

def pca_calc(inpath, pca_num_components, seq_col="sequence", target_col="Rg (nm)"):
    df = pd.read_csv(inpath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {inpath}")

    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' in CSV. Found: {list(df.columns)}")

    # exclude non-embedding columns
    exclude_cols = [seq_col, target_col]
    numeric_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c not in exclude_cols]
    print(f"Using {len(numeric_cols)} numeric columns for PCA.")

    scaler = StandardScaler(with_mean=True)
    pca = PCA(n_components=pca_num_components, svd_solver="randomized", random_state=42)

    X_scaled = scaler.fit_transform(df[numeric_cols])
    X_reduced = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        X_reduced,
        columns=[f"pcaComponent{i+1}" for i in range(pca_num_components)]
    )

    # include sequence and target columns if available
    if seq_col in df.columns:
        pca_df.insert(0, seq_col, df[seq_col].values)
    pca_df[target_col] = df[target_col].values

    Path("data").mkdir(exist_ok=True)
    outpath = f"data/{Path(inpath).stem}_PCA{pca_num_components}_withTarget.csv"
    pca_df.to_csv(outpath, index=False)

    print(f"Saved PCA-reduced CSV → {outpath}")
    print(f"Output columns: {pca_df.columns.tolist()}")

if __name__ == "__main__":
    inpath = "data/protbertEmbeddingsPCA10.csv"
    pca_num_components = 10
    seq_col = "Sequence"
    target_col = "Rg (nm)"

    pca_calc(inpath, pca_num_components, seq_col, target_col)