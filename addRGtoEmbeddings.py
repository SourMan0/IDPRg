import pandas as pd

# === Load both CSV files ===
embeddings_df = pd.read_csv("data/protbertEmbeddingsPCA190.csv")
raw_df = pd.read_csv("data/allRaw.csv")

# === Identify sequence and Rg columns automatically ===
seq_col_emb = [c for c in embeddings_df.columns if "seq" in c.lower()][0]
seq_col_raw = [c for c in raw_df.columns if "seq" in c.lower()][0]
rg_col_raw = [c for c in raw_df.columns if "rg" in c.lower()][0]

# === Merge based on the sequence ===
merged_df = pd.merge(
    embeddings_df,
    raw_df[[seq_col_raw, rg_col_raw]],
    left_on=seq_col_emb,
    right_on=seq_col_raw,
    how="left"
)

# Drop redundant sequence column if duplicated
if seq_col_raw != seq_col_emb:
    merged_df.drop(columns=[seq_col_raw], inplace=True)

# === Overwrite the original embeddings file ===
merged_df.to_csv("data/protbertEmbeddingsPCA190.csv", index=False)

print("Added Rg values to protbertEmbeddings.csv")
print(f"Matched {merged_df[rg_col_raw].notnull().sum()} sequences with Rg values.")