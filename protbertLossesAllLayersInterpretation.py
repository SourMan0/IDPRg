import pandas as pd
import glob
import os
import re

def main():
    # Folder with the ProtBERT loss CSVs
    folder = "protbertLosses/MultipleLayersLosses"
    os.makedirs(folder, exist_ok=True)

    # Step 1: Load all CSVs from the folder
    csv_files = sorted(glob.glob(os.path.join(folder, "protbertLosses*.csv")))
    if not csv_files:
        print(f"No CSV files found in folder '{folder}'.")
        return

    dfs = []
    for i, fpath in enumerate(csv_files, start=1):
        df = pd.read_csv(fpath)

        # Ensure LayerGroup exists
        if "LayerGroup" not in df.columns:
            fname = os.path.basename(fpath).lower()
            m = re.search(r"\b(low|mid|high)\b", fname)
            inferred = m.group(1) if m else "unknown"
            df["LayerGroup"] = inferred

        # Seed = which file this came from
        if "Seed" not in df.columns:
            df["Seed"] = i

        dfs.append(df)

    # Combine all runs
    all_df = pd.concat(dfs, ignore_index=True)

    # Force numeric types (critical for correct sorting!)
    all_df["RMSE Score"] = pd.to_numeric(all_df["RMSE Score"], errors="coerce")
    all_df["Test R2 Score"] = pd.to_numeric(all_df["Test R2 Score"], errors="coerce")

    # Drop rows with NaNs in metrics
    all_df = all_df.dropna(subset=["RMSE Score", "Test R2 Score"])

    # Sanity check
    print(all_df[["RMSE Score", "Test R2 Score"]].dtypes)
    print(f"Total rows after cleaning: {len(all_df)}")

    # ==========================
    # Step 2: Top 10 models per Seed
    # ==========================
    print("===== Top 10 Models per Seed (by RMSE & R2) =====\n")
    for seed, group in all_df.groupby("Seed"):
        top10_rmse = group.sort_values(by="RMSE Score", ascending=True).head(10)
        top10_r2 = group.sort_values(by="Test R2 Score", ascending=False).head(10)
        print(f"--- Seed {seed} ---")
        print("Top 10 RMSE:")
        print(top10_rmse[[
            "Normalization", "Regressing out", "Points", "Model",
            "PCA Components", "LayerGroup", "Regression Type",
            "Test R2 Score", "RMSE Score"
        ]])
        print("\nTop 10 R2:")
        print(top10_r2[[
            "Normalization", "Regressing out", "Points", "Model",
            "PCA Components", "LayerGroup", "Regression Type",
            "Test R2 Score", "RMSE Score"
        ]])
        print("\n")

    # ==========================
    # Step 3: Overall Top 10 rows (per-run)
    # ==========================
    top10_rmse_overall = all_df.sort_values(by="RMSE Score", ascending=True).head(10)
    top10_r2_overall = all_df.sort_values(by="Test R2 Score", ascending=False).head(10)

    # Combined score over rows
    rmse_norm = (all_df["RMSE Score"] - all_df["RMSE Score"].min()) / (
        all_df["RMSE Score"].max() - all_df["RMSE Score"].min()
    )
    r2_norm = (all_df["Test R2 Score"] - all_df["Test R2 Score"].min()) / (
        all_df["Test R2 Score"].max() - all_df["Test R2 Score"].min()
    )
    all_df["Combined Score"] = r2_norm - rmse_norm  # higher = better

    top10_combined = all_df.sort_values(by="Combined Score", ascending=False).head(10)

    print("===== Overall Top 10 RMSE (per-run rows) =====\n")
    print(top10_rmse_overall[[
        "Seed", "Normalization", "Regressing out", "Points", "Model",
        "PCA Components", "LayerGroup", "Regression Type",
        "Test R2 Score", "RMSE Score"
    ]])

    print("\n===== Overall Top 10 R2 (per-run rows) =====\n")
    print(top10_r2_overall[[
        "Seed", "Normalization", "Regressing out", "Points", "Model",
        "PCA Components", "LayerGroup", "Regression Type",
        "Test R2 Score", "RMSE Score"
    ]])

    print("\n===== Overall Top 10 Combined (per-run rows) =====\n")
    print(top10_combined[[
        "Seed", "Normalization", "Regressing out", "Points", "Model",
        "PCA Components", "LayerGroup", "Regression Type",
        "Test R2 Score", "RMSE Score", "Combined Score"
    ]])

    # ==========================
    # Step 4: Config-level performance (mean across seeds)
    # ==========================
    param_cols = [
        "Normalization", "Regressing out", "Points", "Model",
        "PCA Components", "LayerGroup", "Regression Type"
    ]

    config_stats = (
        all_df
        .groupby(param_cols)
        .agg(
            mean_rmse=("RMSE Score", "mean"),
            std_rmse=("RMSE Score", "std"),
            mean_r2=("Test R2 Score", "mean"),
            std_r2=("Test R2 Score", "std"),
            runs=("Seed", "nunique")
        )
        .reset_index()
        .sort_values(by="mean_rmse", ascending=True)
    )

    print("\n===== Top 20 Configs by Mean RMSE (across seeds) =====\n")
    print(config_stats.head(20))

    # ==========================
    # Step 5: Save results
    # ==========================
    top10_rmse_overall.to_csv(os.path.join(folder, "top10_rmse_overall_rows.csv"), index=False)
    top10_r2_overall.to_csv(os.path.join(folder, "top10_r2_overall_rows.csv"), index=False)
    top10_combined.to_csv(os.path.join(folder, "top10_combined_overall_rows.csv"), index=False)
    config_stats.to_csv(os.path.join(folder, "config_stats_mean_rmse_r2.csv"), index=False)
    all_df.sort_values(by="RMSE Score", ascending=True).to_csv(
        os.path.join(folder, "all_rows_sorted_by_rmse.csv"), index=False
    )

    print(f"\nSaved all summaries in folder: {folder}")

if __name__ == "__main__":
    main()