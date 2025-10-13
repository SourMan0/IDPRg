from jax_unirep import get_reps
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Settings
inpath = "data/allRaw.csv"
pca_toggle = False  # In case you want PCA before saving the embeddings
pca_num_components = 200

prot_df = pd.read_csv(inpath)

# CSV cleaning
prot_df = prot_df[prot_df['Protein Sequence'].notnull()]
prot_df = prot_df[prot_df['Protein Sequence'] != '']
prot_df['Protein Sequence'] = prot_df['Protein Sequence'].str.replace('\n', '', regex=True).str.replace(' ', '', regex=False).str.upper()

seqs = prot_df["Protein Sequence"]
avg, h_final, c_final = get_reps(seqs)

# If PCA toggled
if pca_toggle:
    pca = PCA(n_components=pca_num_components, svd_solver="auto", random_state=42)
    scaler = StandardScaler(with_mean=True)

    avg = scaler.fit_transform(avg)
    avg = pca.fit_transform(avg)

prot_df["UniRep Embedding"] = avg.tolist()

prot_df.to_csv("unirepEmbeddings.csv", index=False)