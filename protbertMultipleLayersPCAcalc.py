import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

def pca_calc(inpath, pca_num_components, seq_col="Sequence"):
    """Perform PCA on numeric embedding columns and save reduced CSV."""
    df = pd.read_csv(inpath)
    print(f"\n📂 Loaded {len(df)} rows and {df.shape[1]} columns from {inpath}")

    # --- Validate sequence column ---
    if seq_col not in df.columns:
        raise KeyError(f"Expected '{seq_col}' column in CSV. Found: {list(df.columns)}")

    # --- Identify numeric columns (embedding dimensions) ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric embedding columns found for PCA.")
    print(f"Using {len(numeric_cols)} numeric columns for PCA.")

    # --- Scale and apply PCA ---
    scaler = StandardScaler(with_mean=True)
    X_scaled = scaler.fit_transform(df[numeric_cols])

    pca = PCA(n_components=pca_num_components, svd_solver="randomized", random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    # --- Create PCA-only dataframe ---
    pca_df = pd.DataFrame(
        X_reduced,
        columns=[f"pcaComponent{i+1}" for i in range(pca_num_components)]
    )
    pca_df.insert(0, seq_col, df[seq_col].values)

    # --- Save output ---
    Path("data").mkdir(exist_ok=True)
    base = Path(inpath).stem
    outpath = f"data/{base}_PCA{pca_num_components}.csv"
    pca_df.to_csv(outpath, index=False)

    print(f"✅ Saved PCA-reduced CSV → {outpath}")
    print(f"   Components: {pca_num_components} | Output columns: {len(pca_df.columns)}")


if __name__ == "__main__":
    # Paths for your three ProtBERT layer group files
    layer_files = [
        "data/protbert_embeddings/protbert_low.csv",
        "data/protbert_embeddings/protbert_mid.csv",
        "data/protbert_embeddings/protbert_high.csv"
    ]

    # PCA component counts to test
    pca_values = [10, 20, 50, 100, 190]

    for inpath in layer_files:
        for n_comp in pca_values:
            try:
                pca_calc(inpath, n_comp, seq_col="Experimental Sequence")
            except Exception as e:
                print(f"⚠️ PCA failed for {inpath} with {n_comp} components: {e}")
