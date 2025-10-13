import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ast
from pathlib import Path

# Settings
inpath = "data/unirepEmbeddings.csv"
pca_num_components = 10

df = pd.read_csv(inpath, converters={"UniRep Embedding": ast.literal_eval})
emb = df["UniRep Embedding"].tolist()
print(f"Reducing dimensions from {len(emb[0])} to {pca_num_components}")

# Define PCA and scaler
pca = PCA(n_components=pca_num_components, svd_solver="randomized", random_state=42)
scaler = StandardScaler(with_mean=True)

# Transform embeddings
emb = scaler.fit_transform(emb)
emb = pca.fit_transform(emb)

df["UniRep Embedding"] = emb
df.to_csv(f"data/{Path(inpath).stem}_pca_{pca_num_components}.csv", index=False)