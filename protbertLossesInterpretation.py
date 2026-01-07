import pandas as pd
import glob
import os

def main():
    folder = "protbertLosses/MultipleLayersLosses"
    os.makedirs(folder, exist_ok=True)


    # Step 1: Load all CSVs from the folder
    csv_files = sorted(glob.glob(os.path.join(folder, "protbertLosses*.csv")))
    if not csv_files:
        print(f"No CSV files found in folder '{folder}'.")
        return

    dfs = []
    for i, f in enumerate(csv_files, start=1):
        df = pd.read_csv(f)
        df["Seed"] = i
        dfs.append(df)

    # Combine all runs
    all_df = pd.concat(dfs, ignore_index=True)

    # Step 2: Top 5 models per seed
    print("===== Top 5 Models per Seed =====\n")
    for seed, group in all_df.groupby("Seed"):
        top5_seed = group.sort_values(by="RMSE Score", ascending=True).head(5)
        print(f"--- Seed {seed} ---")
        print(top5_seed[[
            "Normalization",
            "Regressing out",
            "Points",
            "Model",
            "PCA Components",
            "Regression Type",
            "Test R2 Score",
            "RMSE Score"
        ]])
        print("\n")

    # Step 3: Overall Top 5
    all_df_sorted = all_df.sort_values(by="RMSE Score", ascending=True)
    top5_overall = all_df_sorted.head(5)

    print("===== Overall Top 5 Models (Lowest RMSE) =====\n")
    print(top5_overall[[
        "Seed",
        "Normalization",
        "Regressing out",
        "Points",
        "Model",
        "PCA Components",
        "Regression Type",
        "Test R2 Score",
        "RMSE Score"
    ]])

    # Step 4: Parameter consistency check
    param_cols = [
        "Normalization",
        "Regressing out",
        "Points",
        "Model",
        "PCA Components",
        "Regression Type"
    ]

    print("\n===== Parameter Consistency Across Overall Top 5 Models =====\n")
    for col in param_cols:
        unique_vals = top5_overall[col].unique()
        if len(unique_vals) == 1:
            print(f"{col}: consistent ({unique_vals[0]})")
        else:
            print(f"{col}: varies ({unique_vals})")

    # Step 5: Mean RMSE per parameter combination across seeds
    print("\n===== Mean RMSE Across Seeds by Parameter Combination =====\n")
    group_cols = param_cols
    mean_rmse = (
        all_df.groupby(group_cols)["RMSE Score"]
        .mean()
        .reset_index()
        .sort_values(by="RMSE Score", ascending=True)
    )
    print(mean_rmse.head(10))

    # Step 6: Save results in the same folder
    top5_overall_path = os.path.join(folder, "top5_overall_summary.csv")
    mean_rmse_path = os.path.join(folder, "mean_rmse_by_config.csv")
    all_df_sorted_path = os.path.join(folder, "all_sorted_by_rmse.csv")

    top5_overall.to_csv(top5_overall_path, index=False)
    mean_rmse.to_csv(mean_rmse_path, index=False)
    all_df_sorted.to_csv(all_df_sorted_path, index=False)

    print(f"\nSaved results to:\n- {top5_overall_path}\n- {mean_rmse_path}\n- {all_df_sorted_path}")

main()
