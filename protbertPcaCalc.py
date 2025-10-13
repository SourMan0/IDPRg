import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

def pca_calc(inpath, pca_num_components, seq_col="sequence"):
    # --- Load CSV ---
    df = pd.read_csv(inpath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {inpath}")

    # --- Validate sequence column ---
    if seq_col not in df.columns:
        raise KeyError(f"Expected '{seq_col}' column in CSV. Found: {list(df.columns)}")

    # --- Identify numeric columns (embedding dimensions) ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"Using {len(numeric_cols)} numeric columns for PCA.")

    # --- Scale and apply PCA ---
    scaler = StandardScaler(with_mean=True)
    pca = PCA(n_components=pca_num_components, svd_solver="randomized", random_state=42)

    X_scaled = scaler.fit_transform(df[numeric_cols])
    X_reduced = pca.fit_transform(X_scaled)

    # --- Create PCA-only dataframe ---
    pca_df = pd.DataFrame(
        X_reduced,
        columns=[f"pcaComponent{i+1}" for i in range(pca_num_components)]
    )

    # --- Add sequence column ---
    pca_df.insert(0, seq_col, df[seq_col].values)

    # --- Save output ---
    Path("data").mkdir(exist_ok=True)
    outpath = f"data/{Path(inpath).stem}PCA{pca_num_components}.csv"
    pca_df.to_csv(outpath, index=False)

    print(f"Saved PCA-reduced CSV → {outpath}")
    print(f"Output columns: {pca_df.columns.tolist()}")

if __name__ == "__main__":
    inpath = "data/protbertEmbeddings.csv"
    pca_num_components = 190
    seq_col = "Sequence"  # name of the sequence column in your file

    pca_calc(inpath, pca_num_components, seq_col)
