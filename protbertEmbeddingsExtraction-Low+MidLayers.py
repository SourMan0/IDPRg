import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ===============================
# CONFIG
# ===============================
INPUT_FILE = "data/rawData.csv"
EXISTING_HIGH_FILE = "data/protBertEmbeddings2.csv"  # from your previous run
OUTPUT_DIR = "data/protbert_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD DATASET
# ===============================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

print(f"Loading dataset from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}\n")

# ===============================
# DETECT SEQUENCE COLUMN
# ===============================
def is_sequence_column(series):
    """Heuristic: a sequence column will have mostly alphabetic strings (A-Z) of length > 10."""
    try:
        sample_values = series.dropna().astype(str).head(10)
        return sample_values.apply(lambda x: x.isalpha() and len(x) > 10).mean() > 0.5
    except Exception:
        return False

seq_col_candidates = [col for col in df.columns if is_sequence_column(df[col])]
if not seq_col_candidates:
    raise ValueError("Could not detect a sequence column automatically.")
seq_col = seq_col_candidates[0]
print(f"Detected sequence column: '{seq_col}'\n")

# ===============================
# CLEAN SEQUENCES
# ===============================
df = df[df[seq_col].notna() & (df[seq_col].astype(str).str.strip() != "")]
df[seq_col] = df[seq_col].astype(str).str.strip()
print(f"{len(df)} valid sequences after cleaning.\n")

# ===============================
# LOAD PROTBERT MODEL
# ===============================
print("Loading ProtBERT model and tokenizer (Rostlab/prot_bert)...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert", output_hidden_states=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model loaded. Using device: {device}\n")

# ===============================
# FUNCTION: GET MULTI-LAYER EMBEDDINGS (LOW + MID)
# ===============================
def get_protbert_low_mid(sequence: str) -> dict:
    """Compute mean-pooled ProtBERT embeddings for low and mid layer ranges."""
    sequence = sequence.upper().replace(" ", "")
    sequence = " ".join(list(sequence))
    sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X")

    encoded_input = tokenizer(sequence, return_tensors="pt", padding=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.hidden_states  # tuple of 31 tensors (0 = embeddings, 1–30 = layers)

    # Define layer baskets
    low_layers = torch.stack(hidden_states[1:5]).mean(dim=0)   # 1–4
    mid_layers = torch.stack(hidden_states[8:13]).mean(dim=0)  # 8–12

    # Mean-pool across sequence tokens
    low_emb = low_layers.mean(dim=1).squeeze(0).cpu().numpy()
    mid_emb = mid_layers.mean(dim=1).squeeze(0).cpu().numpy()

    return {"low": low_emb, "mid": mid_emb}

# ===============================
# COMPUTE LOW + MID EMBEDDINGS
# ===============================
print("Computing ProtBERT embeddings for low/mid layer baskets...")
low_list, mid_list = [], []
failed_indices = []

for idx, seq in tqdm(enumerate(df[seq_col]), total=len(df)):
    try:
        emb_dict = get_protbert_low_mid(seq)
        low_list.append(emb_dict["low"])
        mid_list.append(emb_dict["mid"])
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        failed_indices.append(idx)
        zero = np.zeros(1024)
        low_list.append(zero)
        mid_list.append(zero)

# ===============================
# SAVE LOW + MID RESULTS
# ===============================
print("\nSaving low/mid embeddings to separate files...")

low_df = pd.concat([df[[seq_col]].reset_index(drop=True),
                    pd.DataFrame(np.vstack(low_list), columns=[f"low_emb_{i}" for i in range(1024)])],
                   axis=1)
low_path = os.path.join(OUTPUT_DIR, "protbert_low.csv")
low_df.to_csv(low_path, index=False)

mid_df = pd.concat([df[[seq_col]].reset_index(drop=True),
                    pd.DataFrame(np.vstack(mid_list), columns=[f"mid_emb_{i}" for i in range(1024)])],
                   axis=1)
mid_path = os.path.join(OUTPUT_DIR, "protbert_mid.csv")
mid_df.to_csv(mid_path, index=False)

print(f"✅ Saved:")
print(f"  • Low-layer embeddings → {low_path}")
print(f"  • Mid-layer embeddings → {mid_path}")

# ===============================
# REUSE EXISTING HIGH-LAYER EMBEDDINGS
# ===============================
if not os.path.exists(EXISTING_HIGH_FILE):
    print(f"\n⚠️ Could not find existing high-layer file: {EXISTING_HIGH_FILE}")
else:
    print(f"\nLoading existing high-layer embeddings from: {EXISTING_HIGH_FILE}")
    high_df = pd.read_csv(EXISTING_HIGH_FILE)

    # Expect columns: "Sequence", emb_0...emb_1023
    # Rename them to match new convention
    high_emb_cols = [c for c in high_df.columns if c.startswith("emb_")]
    rename_map = {c: f"high_{c}" for c in high_emb_cols}
    high_df = high_df.rename(columns=rename_map)

    high_path = os.path.join(OUTPUT_DIR, "protbert_high.csv")
    high_df.to_csv(high_path, index=False)
    print(f"  • High-layer embeddings → {high_path}")

if failed_indices:
    print(f"\n⚠️ {len(failed_indices)} sequences failed to embed (indices: {failed_indices})")
else:
    print("\nAll sequences embedded successfully!")