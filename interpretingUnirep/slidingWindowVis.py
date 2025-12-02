# CELL 1: Sliding-window method visualization from CSV
# Assumes a CSV with at least: "omitted_seq", "delta_rg"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------------------------------
# Load sliding-window results
#   Columns:
#     - omitted_seq : subsequence that was omitted
#     - delta_rg    : Rg_omitted - Rg_original for that window
# -------------------------------------------------

sw_df = pd.read_csv("interpretingUnirep/window_omissions_all.csv")  # <-- change filename if needed

# Keep only rows where omitted subsequence has length 1 (single residue)
sw_df["omitted_seq"] = sw_df["omitted_seq"].astype(str).str.strip()
sw_df = sw_df[sw_df["omitted_seq"].str.len() == 1].copy()

# Rename for clarity
sw_df.rename(columns={"omitted_seq": "residue"}, inplace=True)

print(f"Total single-residue omissions: {len(sw_df)}")
print(sw_df.head())

# -------------------------------------------------
# Aggregate per-residue ΔRg
# -------------------------------------------------

min_count = 30  # drop residues with too few observations

values = defaultdict(list)
for aa, drg in zip(sw_df["residue"], sw_df["delta_rg"]):
    values[aa].append(float(drg))

stats = {}
for aa, arr in values.items():
    arr = np.array(arr, dtype=float)
    if len(arr) < min_count:
        continue
    stats[aa] = {
        "mean":  arr.mean(),
        "std":   arr.std(),
        "count": len(arr),
    }

# -------------------------------------------------
# Plot: mean ΔRg by residue type (sliding-window single-residue deletions)
# -------------------------------------------------
aas   = sorted(stats.keys())
means = [stats[a]["mean"] for a in aas]

plt.figure(figsize=(10, 4))
plt.bar(aas, means)
plt.axhline(0, color="black", linewidth=1)
plt.title("Sliding-window (single-residue) method: Mean ΔRg by residue type")
plt.xlabel("Residue (omitted)")
plt.ylabel("Mean ΔRg (Rg_omitted - Rg_original)")
plt.tight_layout()
plt.show()
