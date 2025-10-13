import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ===============================
# CONFIG
# ===============================
INPUT_FILE = "IDPRg/data/rawData.csv"
OUTPUT_FILE = "protbertEmbeddings.csv"

# Candidate names for sequence column
SEQ_COL_CANDIDATES = [
    "sequence", "Sequence", "Experimental Sequence",
    "Seq", "Protein Sequence", "Amino Acid Sequence"
]

# ===============================
# LOAD DATASET
# ===============================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

print(f"Loading dataset from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows.")
print("\nColumns found in CSV:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head(3))

# ===============================
# DETECT SEQUENCE COLUMN
# ===============================
seq_col = None
for col in SEQ_COL_CANDIDATES:
    if col in df.columns:
        seq_col = col
        break

if seq_col not in df.columns:
    raise ValueError(f"Sequence column '{seq_col}' not found in the dataset.")

print(f"\nUsing column '{seq_col}' for sequences.")

# Drop missing/blank sequences
df = df[df[seq_col].notna() & (df[seq_col].astype(str).str.strip() != "")]
df[seq_col] = df[seq_col].astype(str).str.strip()
print(f"{len(df)} valid sequences after cleaning.\n")

# ===============================
# LOAD PROTBERT MODEL
# ===============================
print("🔹 Loading ProtBERT model and tokenizer (Rostlab/prot_bert)...")
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
    """
    Compute mean-pooled ProtBERT embedding for a single protein sequence.
    """
    # Clean sequence
    sequence = sequence.upper().replace(" ", "")
    sequence = " ".join(list(sequence))
    sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X")

    # Tokenize
    encoded_input = tokenizer(sequence, return_tensors="pt", padding=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Forward pass
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
        print(f"⚠️ Error on row {idx}: {e}")
        failed_indices.append(idx)
        embeddings.append(np.zeros(1024))

# ===============================
# SAVE RESULTS
# ===============================
embeddings = np.vstack(embeddings)
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
emb_df.insert(0, "Sequence", df[seq_col].values)

emb_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved embeddings for {len(df)} sequences to '{OUTPUT_FILE}'")

if failed_indices:
    print(f"⚠️ {len(failed_indices)} sequences failed to embed (indices: {failed_indices})")
