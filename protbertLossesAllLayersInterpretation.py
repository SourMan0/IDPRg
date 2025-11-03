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
        if 'LayerGroup' not in df.columns:
            fname = os.path.basename(fpath).lower()
            m = re.search(r'\b(low|mid|high)\b', fname)
            inferred = m.group(1) if m else "unknown"
            df['LayerGroup'] = inferred
        df["Seed"] = i
        dfs.append(df)

    # Combine all runs
    all_df = pd.concat(dfs, ignore_index=True)

    # Check required columns
    if "RMSE Score" not in all_df.columns or "Test R2 Score" not in all_df.columns:
        raise KeyError("Input CSVs must contain 'RMSE Score' and 'Test R2 Score' columns.")

    # Step 2: Top 10 models per seed for RMSE and R2
    print("===== Top 10 Models per Seed (by RMSE & R2) =====\n")
    for seed, group in all_df.groupby("Seed"):
        top10_rmse = group.sort_values(by="RMSE Score", ascending=True).head(10)
        top10_r2 = group.sort_values(by="Test R2 Score", ascending=False).head(10)
        print(f"--- Seed {seed} ---")
        print("Top 10 RMSE:")
        print(top10_rmse[["Normalization","Regressing out","Points","Model",
                           "PCA Components","LayerGroup","Regression Type",
                           "Test R2 Score","RMSE Score"]])
        print("\nTop 10 R2:")
        print(top10_r2[["Normalization","Regressing out","Points","Model",
                         "PCA Components","LayerGroup","Regression Type",
                         "Test R2 Score","RMSE Score"]])
        print("\n")

    # Step 3: Overall Top 10
    top10_rmse_overall = all_df.sort_values(by="RMSE Score", ascending=True).head(10)
    top10_r2_overall = all_df.sort_values(by="Test R2 Score", ascending=False).head(10)

    # Combined ranking: normalize RMSE and R2 to 0-1 scale and sum
    rmse_norm = (all_df["RMSE Score"] - all_df["RMSE Score"].min()) / (all_df["RMSE Score"].max() - all_df["RMSE Score"].min())
    r2_norm = (all_df["Test R2 Score"] - all_df["Test R2 Score"].min()) / (all_df["Test R2 Score"].max() - all_df["Test R2 Score"].min())
    all_df["Combined Score"] = r2_norm - rmse_norm  # maximize R2, minimize RMSE
    top10_combined = all_df.sort_values(by="Combined Score", ascending=False).head(10)

    print("===== Overall Top 10 RMSE =====\n")
    print(top10_rmse_overall[["Seed","Normalization","Regressing out","Points","Model",
                               "PCA Components","LayerGroup","Regression Type",
                               "Test R2 Score","RMSE Score"]])

    print("\n===== Overall Top 10 R2 =====\n")
    print(top10_r2_overall[["Seed","Normalization","Regressing out","Points","Model",
                             "PCA Components","LayerGroup","Regression Type",
                             "Test R2 Score","RMSE Score"]])

    print("\n===== Overall Top 10 Combined (R2 & RMSE) =====\n")
    print(top10_combined[["Seed","Normalization","Regressing out","Points","Model",
                           "PCA Components","LayerGroup","Regression Type",
                           "Test R2 Score","RMSE Score","Combined Score"]])

    # Step 4: Parameter consistency across top 10 combined
    param_cols = ["Normalization","Regressing out","Points","Model","PCA Components","LayerGroup","Regression Type"]
    print("\n===== Parameter Consistency Across Overall Top 10 Combined =====\n")
    for col in param_cols:
        unique_vals = top10_combined[col].unique()
        if len(unique_vals) == 1:
            print(f"{col}: consistent ({unique_vals[0]})")
        else:
            print(f"{col}: varies ({unique_vals})")

    # Step 5: Mean RMSE per parameter combination across seeds
    print("\n===== Mean RMSE Across Seeds by Parameter Combination =====\n")
    mean_rmse = (
        all_df.groupby(param_cols)["RMSE Score"]
        .mean()
        .reset_index()
        .sort_values(by="RMSE Score", ascending=True)
    )
    print(mean_rmse.head(20))

    # Step 6: Save results
    top10_rmse_overall.to_csv(os.path.join(folder, "top10_rmse_overall.csv"), index=False)
    top10_r2_overall.to_csv(os.path.join(folder, "top10_r2_overall.csv"), index=False)
    top10_combined.to_csv(os.path.join(folder, "top10_combined_overall.csv"), index=False)
    mean_rmse.to_csv(os.path.join(folder, "mean_rmse_by_config.csv"), index=False)
    all_df.sort_values(by="RMSE Score", ascending=True).to_csv(os.path.join(folder, "all_sorted_by_rmse.csv"), index=False)

    print(f"\nSaved all summaries in folder: {folder}")

if __name__ == "__main__":
    main()