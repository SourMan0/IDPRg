import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def main():
    folder = "protbertLosses"
    os.makedirs(folder, exist_ok=True)

    # Load all CSVs
    csv_files = sorted(glob.glob(os.path.join(folder, "protbertLosses*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{folder}'.")
        return

    dfs = []
    for i, f in enumerate(csv_files, start=1):
        df = pd.read_csv(f)
        df["Seed"] = i
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # --- Plot 1: RMSE distribution per seed ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Seed", y="RMSE Score", data=all_df, palette="viridis")
    plt.title("RMSE Distribution per Seed")
    plt.ylabel("RMSE Score")
    plt.xlabel("Seed")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "rmse_distribution_per_seed.png"), dpi=300)
    plt.close()

    # --- Plot 2: R² vs RMSE scatterplot ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=all_df,
        x="RMSE Score",
        y="Test R2 Score",
        hue="Seed",
        palette="tab10",
        s=60,
        edgecolor="k"
    )
    plt.title("R² vs RMSE (Colored by Seed)")
    plt.xlabel("RMSE Score (Lower is Better)")
    plt.ylabel("Test R² Score (Higher is Better)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "r2_vs_rmse_scatter.png"), dpi=300)
    plt.close()

    # --- Plot 3: Mean RMSE by Regression Type and PCA Components ---
    mean_rmse = (
        all_df.groupby(["Regression Type", "PCA Components"])["RMSE Score"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=mean_rmse,
        x="PCA Components",
        y="RMSE Score",
        hue="Regression Type",
        palette="mako"
    )
    plt.title("Mean RMSE by Regression Type and PCA Components")
    plt.xlabel("PCA Components")
    plt.ylabel("Mean RMSE Score")
    plt.legend(title="Regression Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "mean_rmse_by_regression_pca.png"), dpi=300)
    plt.close()

    print("Saved plots:")
    print(" - rmse_distribution_per_seed.png")
    print(" - r2_vs_rmse_scatter.png")
    print(" - mean_rmse_by_regression_pca.png")

main()
