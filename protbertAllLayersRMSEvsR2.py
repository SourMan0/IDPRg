import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Load your combined loss file ===
# replace with the actual path if needed
loss_path = Path("protbertLosses/MultipleLayersLosses/all_sorted_by_rmse.csv")
df = pd.read_csv(loss_path)

# Optional sanity check
print(df.head())
print("\nUnique LayerGroups:", df["LayerGroup"].unique())
print("Unique PCA Components:", sorted(df["PCA Components"].unique()))

# === Plot 1: R² vs PCA components for each layer ===
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df,
    x="PCA Components",
    y="Test R2 Score",
    hue="LayerGroup",
    style="Points",  # All vs Inliers
    markers=True,
    dashes=False,
    ci="sd"
)

plt.title("ProtBERT Performance by Layer Group")
plt.xlabel("PCA Components")
plt.ylabel("Test $R^2$ Score")
plt.legend(title="Layer / Data Type")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === Plot 2 (optional): RMSE vs PCA components ===
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df,
    x="PCA Components",
    y="RMSE Score",
    hue="LayerGroup",
    style="Points",
    markers=True,
    dashes=False,
    ci="sd"
)

plt.title("RMSE Across ProtBERT Layers")
plt.xlabel("PCA Components")
plt.ylabel("RMSE (lower is better)")
plt.legend(title="Layer / Data Type")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
