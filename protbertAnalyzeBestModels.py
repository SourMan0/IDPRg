#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    sns.set(style="whitegrid")

    csv_path = "/Users/zmrao/Desktop/Mofrad_Lab/IDPRg/protbertLossesOLD/MultipleLayersLosses/loss_plots/merged_losses_clean.csv"
    out_dir = "analysis_plots"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # --- Focus on the main setting you care about ---
    df = df.copy()
    df = df[df["Points"] == "Inliers"]
    df = df[df["Normalization"] == "Rg norm w/0.5"]

    # Ensure proper categories
    df["PCA Components"] = df["PCA Components"].astype(int).astype(str)
    df["LayerGroup"] = df["LayerGroup"].astype(str)

    # =====================================================================
    # 1) Full regression type comparison
    # =====================================================================
    reg_summary = (
        df.groupby("Regression Type")["RMSE Score"]
          .agg(["mean", "std", "min", "count"])
          .reset_index()
          .sort_values("mean")
    )

    print("\n=== Regression type summary (sorted by mean RMSE) ===")
    print(reg_summary)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=reg_summary,
        x="Regression Type",
        y="mean",
        order=reg_summary["Regression Type"],
    )
    plt.ylabel("Mean RMSE Score")
    plt.xlabel("Regression Type")
    plt.title("Mean RMSE by Regression Type (Inliers, Rg norm w/0.5)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_rmse_by_regression_type.png"), dpi=300)
    plt.close()

    # =====================================================================
    # 2) PCA comparison for selected regression types (boxplot)
    # =====================================================================
    best_regs = ["Kernel Ridge", "GPR", "Lasso", "Ridge"]
    df_best_regs = df[df["Regression Type"].isin(best_regs)].copy()

    pca_order = sorted(df_best_regs["PCA Components"].unique(), key=lambda x: int(x))

    # For logging: mean RMSE per (Regression Type, PCA)
    pca_summary = (
        df_best_regs.groupby(["Regression Type", "PCA Components"])["RMSE Score"]
                    .mean()
                    .reset_index()
    )
    print("\n=== PCA summary for selected regressions (mean RMSE) ===")
    print(pca_summary.sort_values("RMSE Score"))

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_best_regs,
        x="PCA Components",
        y="RMSE Score",
        order=pca_order,
        hue="Regression Type",
        linewidth=1.2,
    )
    plt.xlabel("PCA Components", fontsize=12)
    plt.ylabel("RMSE Score", fontsize=12)
    plt.title("RMSE Distribution Across PCA Components\n(Kernel Ridge, GPR, Lasso, Ridge)", fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title="Regression Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_by_pca_boxplot_selected_regs.png"), dpi=300)
    plt.close()

    # =====================================================================
    # 2b) Effect of PCA x LayerGroup on RMSE (heatmap)
    # =====================================================================
    effect_df = (
        df_best_regs
        .groupby(["LayerGroup", "PCA Components"])["RMSE Score"]
        .mean()
        .reset_index()
    )

    # define consistent layer order
    layer_order = [g for g in ["low", "mid", "high"] if g in effect_df["LayerGroup"].unique()]
    effect_df["LayerGroup"] = pd.Categorical(effect_df["LayerGroup"], categories=layer_order, ordered=True)

    heatmap_data = (
        effect_df
        .pivot(index="LayerGroup", columns="PCA Components", values="RMSE Score")
        .loc[layer_order, pca_order]  # ensure proper ordering
    )

    print("\n=== Mean RMSE by LayerGroup x PCA Components ===")
    print(heatmap_data)

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        cbar_kws={"label": "Mean RMSE"},
    )
    plt.xlabel("PCA Components", fontsize=12)
    plt.ylabel("ProtBERT Layer Group", fontsize=12)
    plt.title("Effect of PCA and Layer Group on RMSE\n(Kernel Ridge, GPR, Lasso, Ridge)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_heatmap_pca_layergroup_selected_regs.png"), dpi=300)
    plt.close()

    # =====================================================================
    # 3) Layer group comparison using best PCA values (100/190)
    # =====================================================================
    best_pcas = ["100", "190"]
    df_core = df_best_regs[df_best_regs["PCA Components"].isin(best_pcas)].copy()

    layer_summary = (
        df_core.groupby("LayerGroup")["RMSE Score"]
               .agg(["mean", "std", "min", "count"])
               .reset_index()
               .sort_values("mean")
    )

    print("\n=== LayerGroup summary (selected regressions, PCA 100/190) ===")
    print(layer_summary)

    plt.figure(figsize=(6, 5))
    sns.barplot(
        data=layer_summary,
        x="LayerGroup",
        y="mean",
        order=layer_summary["LayerGroup"],
    )
    plt.xlabel("ProtBERT Layer Group")
    plt.ylabel("Mean RMSE Score")
    plt.title("Mean RMSE by Layer Group\n(Kernel Ridge + GPR + Lasso + Ridge, PCA 100/190)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_rmse_by_layergroup_core_models.png"), dpi=300)
    plt.close()

    # RMSE distribution boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df_core,
        x="LayerGroup",
        y="RMSE Score",
        order=["low", "mid", "high"] \
            if set(df_core["LayerGroup"]) >= {"low", "mid", "high"} else None,
    )
    plt.xlabel("ProtBERT Layer Group")
    plt.ylabel("RMSE Score")
    plt.title("RMSE Distribution by Layer Group\n(Kernel Ridge + GPR + Lasso + Ridge, PCA 100/190)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_boxplot_by_layergroup_core_models.png"), dpi=300)
    plt.close()

    # =====================================================================
    # 4) Scatter: RMSE vs PCA for selected regressions
    # =====================================================================
    plt.figure(figsize=(9, 5))
    sns.stripplot(
        data=df_best_regs,
        x="PCA Components",
        y="RMSE Score",
        hue="Regression Type",
        order=pca_order,
        dodge=True,
        alpha=0.7,
    )
    plt.xlabel("PCA Components")
    plt.ylabel("RMSE Score")
    plt.title("RMSE by PCA for Kernel Ridge + GPR + Lasso + Ridge")
    plt.legend(title="Regression Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_by_pca_strip_selected_regs.png"), dpi=300)
    plt.close()

    # =====================================================================
    # 5) Ranked table
    # =====================================================================
    ranked = (
        df.sort_values("RMSE Score", ascending=True)
          .reset_index(drop=True)
    )
    top_n = 50
    ranked_path = os.path.join(out_dir, f"top_{top_n}_models_by_rmse.csv")
    ranked.head(top_n).to_csv(ranked_path, index=False)
    print(f"\nSaved ranked top-{top_n} models to: {ranked_path}")

    print("\nFigures saved to:", out_dir)

if __name__ == "__main__":
    main()
