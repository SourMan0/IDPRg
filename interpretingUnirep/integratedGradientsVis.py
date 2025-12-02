# CELL 2: Integrated Gradients visualization

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load per-AA IG scores from earlier IG pipeline
# (ensure you've run the IG aggregation and saved aa_ig_scores.csv)
# -------------------------------------------------
aa_df = pd.read_csv("interpretingUnirep/aa_ig_scores_oldpolyA.csv")

# aa_df columns should be: ["channel_index", "aa_label", "ig_score"]
print(aa_df.head())

# -------------------------------------------------
# Filter to canonical 20 amino acids
# -------------------------------------------------
canonical_aas = list("ACDEFGHIKLMNPQRSTVWY")  # standard residues

aa_df_can = aa_df[aa_df["aa_label"].isin(canonical_aas)].copy()

# Ensure consistent ordering
aa_df_can["aa_label"] = pd.Categorical(aa_df_can["aa_label"],
                                       categories=canonical_aas,
                                       ordered=True)
aa_df_can = aa_df_can.sort_values("aa_label")

# -------------------------------------------------
# Plot: global IG score per amino acid type
# -------------------------------------------------
plt.figure(figsize=(10, 4))
plt.bar(aa_df_can["aa_label"], aa_df_can["ig_score"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Integrated Gradients: Global IG score by residue type")
plt.xlabel("Residue")
plt.ylabel("Total IG score (relative to baseline)")
plt.tight_layout()
plt.show()
