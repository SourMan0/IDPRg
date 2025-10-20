import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ===============================
# CONFIG
# ===============================
INPUT_FILE = "data/inliersNormalizedWithAll.csv"
OUTPUT_FILE = "data/protbertEmbeddings-inliersNormalizedWithAll.csv"

# ===============================
# LOAD DATASET
# ===============================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

print(f"Loading dataset from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}\n")

if df.shape[1] != 2:
    raise ValueError(f"Expected exactly 2 columns (sequence and Rg), but found {df.shape[1]}.")

# ===============================
# DETECT SEQUENCE AND TARGET COLUMNS
# ===============================
def is_sequence_column(series):
    """Heuristic: a sequence column will have mostly alphabetic strings (A-Z) of length > 10."""
    try:
        sample_values = series.dropna().astype(str).head(10)
        return sample_values.apply(lambda x: x.isalpha() and len(x) > 10).mean() > 0.5
    except Exception:
        return False

col1, col2 = df.columns
if is_sequence_column(df[col1]):
    seq_col, rg_col = col1, col2
elif is_sequence_column(df[col2]):
    seq_col, rg_col = col2, col1
else:
    raise ValueError("Could not automatically detect which column contains amino acid sequences.")

print(f"Detected sequence column: '{seq_col}'")
print(f"Detected target (Rg) column: '{rg_col}'\n")

# Clean sequences
df = df[df[seq_col].notna() & (df[seq_col].astype(str).str.strip() != "")]
df[seq_col] = df[seq_col].astype(str).str.strip()
print(f"{len(df)} valid sequences after cleaning.\n")

# ===============================
# LOAD PROTBERT MODEL
# ===============================
print("Loading ProtBERT model and tokenizer (Rostlab/prot_bert)...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model loaded. Using device: {device}\n")

# ===============================
# FUNCTION: GET EMBEDDING
# ===============================
def get_protbert_embedding(sequence: str) -> np.ndarray:
    """Compute mean-pooled ProtBERT embedding for a single protein sequence."""
    sequence = sequence.upper().replace(" ", "")
    sequence = " ".join(list(sequence))
    sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X")

    encoded_input = tokenizer(sequence, return_tensors="pt", padding=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state.squeeze(0)
        embedding = hidden_states.mean(dim=0).cpu().numpy()
    return embedding

# ===============================
# COMPUTE EMBEDDINGS
# ===============================
print("Computing ProtBERT embeddings...")
embeddings = []
failed_indices = []

for idx, seq in tqdm(enumerate(df[seq_col]), total=len(df)):
    try:
        emb = get_protbert_embedding(seq)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        failed_indices.append(idx)
        embeddings.append(np.zeros(1024))

# ===============================
# SAVE RESULTS
# ===============================
embeddings = np.vstack(embeddings)
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
emb_df.insert(0, "Sequence", df[seq_col].values)
emb_df["Rg (nm)"] = df[rg_col].values

emb_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved embeddings for {len(df)} sequences to '{OUTPUT_FILE}'")

if failed_indices:
    print(f"Warning: {len(failed_indices)} sequences failed to embed (indices: {failed_indices})")