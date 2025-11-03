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
    out_dir = Path("data/protbert_pca")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(inpath).stem
    outpath = out_dir / f"{base}_PCA{pca_num_components}.csv"
    pca_df.to_csv(outpath, index=False)

    print(f"✅ Saved PCA-reduced CSV → {outpath}")
    print(f"   Components: {pca_num_components} | Output columns: {len(pca_df.columns)}")


if __name__ == "__main__":
    # Paths for your three ProtBERT layer group files
    layer_files = {
        "low": "data/protbert_embeddings/protbert_low.csv",
        "mid": "data/protbert_embeddings/protbert_mid.csv",
        "high": "data/protbert_embeddings/protbert_high.csv"
    }

    # PCA component counts to test
    pca_values = [10, 20, 50, 100, 190]

    for layer_name, inpath in layer_files.items():
        if not Path(inpath).exists():
            print(f"⚠️ File for layer '{layer_name}' not found: {inpath}")
            continue

        print(f"\n🔹 Processing {layer_name.upper()} layer embeddings...")
        for n_comp in pca_values:
            try:
                pca_calc(inpath, n_comp, seq_col="Sequence")
            except Exception as e:
                print(f"⚠️ PCA failed for {layer_name} ({n_comp} comps): {e}")
