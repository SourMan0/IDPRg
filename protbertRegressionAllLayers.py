import os
import re
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from doAllRegressions import evaluate_models_rmse


# ======================
# Helpers
# ======================
def canon_col(s: str) -> str:
    """Canonicalize a column name so minor formatting differences don't break selection."""
    s = str(s).strip().lower()
    # remove units and parenthetical notes
    s = re.sub(r"\([^)]*\)", "", s)         # remove "(nm)" etc
    s = re.sub(r"\s+", " ", s).strip()      # collapse whitespace
    return s

def build_col_map(df: pd.DataFrame):
    """Map canonical column name -> original column name (first occurrence)."""
    m = {}
    for c in df.columns:
        key = canon_col(c)
        if key not in m:
            m[key] = c
    return m

def resolve_target_col(df: pd.DataFrame, desired: str) -> str:
    """Find the best matching actual column name in df for 'desired'."""
    cmap = build_col_map(df)
    key = canon_col(desired)
    if key in cmap:
        return cmap[key]

    # Try some extra normalizations if desired has special characters
    key2 = re.sub(r"[^\w\s]", " ", key)
    key2 = re.sub(r"\s+", " ", key2).strip()
    if key2 in cmap:
        return cmap[key2]

    # If still missing, show helpful candidates
    candidates = sorted(df.columns.tolist())
    raise KeyError(
        f"Target column not found.\n"
        f"  Wanted: {desired!r}\n"
        f"  Canon:  {key!r}\n"
        f"  Available columns include:\n"
        f"    - " + "\n    - ".join(candidates[:40]) +
        ("" if len(candidates) <= 40 else f"\n    ... (+{len(candidates)-40} more)")
    )

def find_seq_col(df: pd.DataFrame) -> str:
    if "Sequence" in df.columns:
        return "Sequence"
    if "Experimental Sequence" in df.columns:
        return "Experimental Sequence"
    raise KeyError(f"No sequence column found. Columns: {list(df.columns)}")

def find_embedding_cols(df: pd.DataFrame):
    # Prefer pcaComponent* if present
    pca_cols = [c for c in df.columns if re.match(r"^pcaComponent\d+$", str(c))]
    if pca_cols:
        # sort numerically
        pca_cols.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        return pca_cols

    # Otherwise accept pc1, pc2...
    pc_cols = [c for c in df.columns if re.match(r"^pc\d+$", str(c))]
    if pc_cols:
        pc_cols.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        return pc_cols

    # Otherwise: numeric columns except the sequence column
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    return num_cols


# ======================
# OUTPUT DIR
# ======================
os.makedirs("protbertLosses/MultipleLayersLosses", exist_ok=True)

# ======================
# LOAD LABELS
# ======================
all_points_df = pd.read_csv("training/all_points.csv")
inliers_df = pd.read_csv("training/inliers.csv")

# strip whitespace just in case
all_points_df.columns = all_points_df.columns.str.strip()
inliers_df.columns = inliers_df.columns.str.strip()

all_sequences = all_points_df["Sequence"].to_list()
inlier_sequences = inliers_df["Sequence"].to_list()

# ======================
# PCA FILES (EDIT THIS PATH)
# Put your attached file(s) in this folder, or point to where they are.
# ======================
data_path = Path("data/protbert_embeddings")  # <- change if needed
pca_files = sorted(data_path.glob("protbert_*_PCA*.csv"))

if not pca_files:
    raise FileNotFoundError(
        f"No PCA embedding files found in {data_path.resolve()}.\n"
        "Expected files like protbert_high_PCA10.csv"
    )

print(f"Found {len(pca_files)} PCA embedding files:")
for f in pca_files:
    print(" -", f.name)

# ======================
# LABEL HEADER NAMES YOU WANT
# (these are your "desired" names, but we will resolve to actual df columns safely)
# ======================
labelHeaders = [
    'Sequence',
    'Rg (nm)',

    'Rg normalized w/0.421',                 # was 0.427
    'Rg normalized w/0.5 (nm)',
    'Rg normalized w/0.406 (nm)',            # was 0.418

    'Rg w/pH regressed out',
    'Rg normalized w/0.421 w/pH regressed out',
    'Rg normalized w/0.5 w/pH regressed out',
    'Rg normalized w/0.406 w/pH regressed out',

    'Rg w/buffer regressed out',
    'Rg normalized w/0.421 w/buffer regressed out',
    'Rg normalized w/0.5 w/buffer regressed out',
    'Rg normalized w/0.406 w/buffer regressed out',

    'Rg w/experimental pH regressed out',
    'Rg normalized w/0.421 w/experimental pH regressed out',
    'Rg normalized w/0.5 w/experimental pH regressed out',
    'Rg normalized w/0.406 w/experimental pH regressed out',

    'Rg w/experimental buffer regressed out',
    'Rg normalized w/0.421 w/experimental buffer regressed out',
    'Rg normalized w/0.5 w/experimental buffer regressed out',
    'Rg normalized w/0.406 w/experimental buffer regressed out'
]


labelSplits = [
    ['Rg w/no norm', 'No regr out'],
    ['Rg norm w/0.421', 'No regr out'],   # was 0.427
    ['Rg norm w/0.5', 'No regr out'],
    ['Rg norm w/0.406', 'No regr out'],   # was 0.418
    ['Rg w/no norm', 'pH regr out'],
    ['Rg norm w/0.421', 'pH regr out'],
    ['Rg norm w/0.5', 'pH regr out'],
    ['Rg norm w/0.406', 'pH regr out'],
    ['Rg w/no norm', 'buffer regr out'],
    ['Rg norm w/0.421', 'buffer regr out'],
    ['Rg norm w/0.5', 'buffer regr out'],
    ['Rg norm w/0.406', 'buffer regr out'],
    ['Rg w/no norm', 'expr pH only regr out'],
    ['Rg norm w/0.421', 'expr pH only regr out'],
    ['Rg norm w/0.5', 'expr pH only regr out'],
    ['Rg norm w/0.406', 'expr pH only regr out'],
    ['Rg w/no norm', 'expr buffer only regr out'],
    ['Rg norm w/0.421', 'expr buffer only regr out'],
    ['Rg norm w/0.5', 'expr buffer only regr out'],
    ['Rg norm w/0.406', 'expr buffer only regr out']
]

# Pre-resolve label columns once (so we fail early with a good message)
resolved_all = {}
resolved_inl = {}
for i in range(20):
    desired = labelHeaders[i + 1]
    resolved_all[desired] = resolve_target_col(all_points_df, desired)
    resolved_inl[desired] = resolve_target_col(inliers_df, desired)

print("\nResolved label columns (desired -> actual):")
for desired, actual in resolved_all.items():
    if desired != actual:
        print(f" - {desired}  ->  {actual}")

# ======================
# RUN
# ======================
for idx, seed in enumerate([43, 44, 45, 46], start=1):
    outfile = f"protbertLosses/MultipleLayersLosses/protbertLosses{idx}.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Normalization", "Regressing out", "Points", "Model", "LayerGroup",
            "PCA Components", "Regression Type", "Test Split", "Test R2 Score", "RMSE Score"
        ])

        for file in pca_files:
            filename = file.name

            # Parse layer + PCA number
            layer_match = re.search(r"protbert_(low|mid|high)", filename)
            pca_match = re.search(r"PCA(\d+)", filename)
            layer_group = layer_match.group(1) if layer_match else "unknown"
            pca_num = int(pca_match.group(1)) if pca_match else -1

            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()

            seq_col = find_seq_col(df)
            emb_cols = find_embedding_cols(df)

            # Rename embedding columns uniformly: pc1, pc2, ...
            rename_map = {c: f"pc{i+1}" for i, c in enumerate(emb_cols)}
            df = df.rename(columns=rename_map)

            X_cols = [f"pc{i+1}" for i in range(len(emb_cols))]

            # Filter sequences
            df_all = df[df[seq_col].isin(all_sequences)].copy()
            df_inl = df[df[seq_col].isin(inlier_sequences)].copy()

            print(f"Running PCA{pca_num} | {layer_group}: {df_all.shape[0]} all, {df_inl.shape[0]} inliers")

            for label_idx, ls in enumerate(labelSplits):
                desired_target = labelHeaders[label_idx + 1]
                target_all = resolved_all[desired_target]
                target_inl = resolved_inl[desired_target]

                # Merge labels onto embeddings
                merged_all = df_all.merge(
                    all_points_df[["Sequence", target_all]],
                    left_on=seq_col,
                    right_on="Sequence",
                    how="inner"
                )
                merged_inl = df_inl.merge(
                    inliers_df[["Sequence", target_inl]],
                    left_on=seq_col,
                    right_on="Sequence",
                    how="inner"
                )

                X_all = merged_all[X_cols].to_numpy()
                y_all = merged_all[target_all].to_numpy()

                X_inl = merged_inl[X_cols].to_numpy()
                y_inl = merged_inl[target_inl].to_numpy()

                losses_all = evaluate_models_rmse(X_all, y_all, seed)
                losses_inl = evaluate_models_rmse(X_inl, y_inl, seed)

                for loss in losses_all:
                    writer.writerow([
                        ls[0], ls[1], "All", "ProtBERT", layer_group, pca_num,
                        loss[0], loss[1], loss[2], loss[3]
                    ])

                for loss in losses_inl:
                    writer.writerow([
                        ls[0], ls[1], "Inliers", "ProtBERT", layer_group, pca_num,
                        loss[0], loss[1], loss[2], loss[3]
                    ])

print("\n✅ All ProtBERT PCA regressions completed. Results saved to protbertLosses/MultipleLayersLosses/protbertLosses*.csv")