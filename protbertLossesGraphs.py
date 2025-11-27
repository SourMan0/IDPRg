#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_loss_plots.py
------------------
Creates diagnostic/interpretation plots for ProtBERT->PCA->Regression
loss results (R2 and RMSE) across multiple configurations and seeds.

UPDATED:
- Adds explicit LayerGroup accuracy comparison plots (low vs mid vs high).

Usage:
  python make_loss_plots.py \
    --data_dir "/Users/zmrao/Desktop/Mofrad_Lab/IDPRg/protbertLosses/MultipleLayersLosses" \
    --out_dir loss_plots \
    --top_n 20
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def read_all_csvs(data_dir):
    """Read all CSVs in data_dir, return dict(name->df)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    dfs = {}
    for p in paths:
        try:
            df = pd.read_csv(p)
            if len(df) == 0:
                continue
            dfs[os.path.basename(p)] = df
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    return dfs


def standardize_columns(df):
    """Make column names consistent and strip whitespace."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "PCA components": "PCA Components",
        "PCA Component": "PCA Components",
        "Regression type": "Regression Type",
        "Test R2": "Test R2 Score",
        "RMSE": "RMSE Score",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    return df


def concat_loss_tables(dfs):
    """Concatenate CSVs that look like per-run losses."""
    loss_like = []
    for name, df in dfs.items():
        df = standardize_columns(df)
        if ("Test R2 Score" in df.columns) and ("RMSE Score" in df.columns):
            df["_source_file"] = name
            loss_like.append(df)

    if not loss_like:
        raise ValueError("No loss-like CSVs found (need Test R2 Score + RMSE Score).")

    return pd.concat(loss_like, ignore_index=True)


def ensure_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def savefig(out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[SAVED] {path}")


# -----------------------------
# Existing Plotters
# -----------------------------
def plot_rmse_r2_by_pca(df, out_dir):
    """Mean RMSE and R2 vs PCA Components, grouped by Regression Type."""
    if "PCA Components" not in df.columns or "Regression Type" not in df.columns:
        return

    g = df.dropna(subset=["PCA Components", "Regression Type", "RMSE Score", "Test R2 Score"])
    g["PCA Components"] = g["PCA Components"].astype(int)

    agg = (
        g.groupby(["Regression Type", "PCA Components"])
        .agg(rmse_mean=("RMSE Score", "mean"),
             rmse_std=("RMSE Score", "std"),
             r2_mean=("Test R2 Score", "mean"),
             r2_std=("Test R2 Score", "std"))
        .reset_index()
        .sort_values("PCA Components")
    )

    reg_types = agg["Regression Type"].unique()

    # RMSE vs PCA
    plt.figure()
    for rt in reg_types:
        sub = agg[agg["Regression Type"] == rt]
        plt.plot(sub["PCA Components"], sub["rmse_mean"], marker="o", label=rt)
        plt.fill_between(
            sub["PCA Components"],
            sub["rmse_mean"] - sub["rmse_std"].fillna(0),
            sub["rmse_mean"] + sub["rmse_std"].fillna(0),
            alpha=0.15,
        )
    plt.xlabel("PCA Components")
    plt.ylabel("RMSE (mean ± std)")
    plt.title("RMSE vs PCA Components by Regression Type")
    plt.legend(fontsize=8)
    savefig(out_dir, "rmse_vs_pca_by_regression.png")

    # R2 vs PCA
    plt.figure()
    for rt in reg_types:
        sub = agg[agg["Regression Type"] == rt]
        plt.plot(sub["PCA Components"], sub["r2_mean"], marker="o", label=rt)
        plt.fill_between(
            sub["PCA Components"],
            sub["r2_mean"] - sub["r2_std"].fillna(0),
            sub["r2_mean"] + sub["r2_std"].fillna(0),
            alpha=0.15,
        )
    plt.xlabel("PCA Components")
    plt.ylabel("Test R2 (mean ± std)")
    plt.title("R2 vs PCA Components by Regression Type")
    plt.legend(fontsize=8)
    savefig(out_dir, "r2_vs_pca_by_regression.png")


def plot_box_rmse_r2_by_regtype(df, out_dir):
    """Box plots for RMSE and R2 grouped by Regression Type."""
    if "Regression Type" not in df.columns:
        return

    g = df.dropna(subset=["Regression Type", "RMSE Score", "Test R2 Score"])
    reg_types = sorted(g["Regression Type"].unique())

    plt.figure()
    data = [g[g["Regression Type"] == rt]["RMSE Score"].values for rt in reg_types]
    plt.boxplot(data, labels=reg_types, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("RMSE")
    plt.title("RMSE Distribution by Regression Type")
    savefig(out_dir, "box_rmse_by_regression.png")

    plt.figure()
    data = [g[g["Regression Type"] == rt]["Test R2 Score"].values for rt in reg_types]
    plt.boxplot(data, labels=reg_types, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test R2")
    plt.title("R2 Distribution by Regression Type")
    savefig(out_dir, "box_r2_by_regression.png")


def plot_r2_vs_rmse_scatter(df, out_dir):
    """Scatter of R2 vs RMSE with best configs highlighted."""
    g = df.dropna(subset=["RMSE Score", "Test R2 Score"])

    plt.figure()
    plt.scatter(g["RMSE Score"], g["Test R2 Score"], s=8, alpha=0.3)
    plt.xlabel("RMSE")
    plt.ylabel("Test R2")
    plt.title("R2 vs RMSE (all runs/configs)")

    r2_thr = np.nanpercentile(g["Test R2 Score"], 95)
    rmse_thr = np.nanpercentile(g["RMSE Score"], 5)
    good = g[(g["Test R2 Score"] >= r2_thr) & (g["RMSE Score"] <= rmse_thr)]
    if len(good) > 0:
        plt.scatter(good["RMSE Score"], good["Test R2 Score"], s=25, alpha=0.9)
    savefig(out_dir, "scatter_r2_vs_rmse.png")


def plot_heatmap_mean_rmse(df, out_dir, row="LayerGroup", col="PCA Components"):
    """Heatmap of mean RMSE for row x col."""
    if row not in df.columns or col not in df.columns:
        return

    g = df.dropna(subset=[row, col, "RMSE Score"]).copy()
    g[col] = pd.to_numeric(g[col], errors="coerce")

    pivot = g.pivot_table(index=row, columns=col, values="RMSE Score", aggfunc="mean")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return

    plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.colorbar(label="Mean RMSE")
    plt.title(f"Mean RMSE Heatmap: {row} x {col}")
    savefig(out_dir, f"heatmap_mean_rmse_{row}_x_{col}.png")


def plot_top_configs(df, out_dir, top_n=20):
    """Bar plot of top configs by highest R2 and lowest RMSE."""
    key_cols = [c for c in [
        "Normalization", "Regressing out", "Points", "LayerGroup",
        "PCA Components", "Regression Type", "Test Split"
    ] if c in df.columns]

    g = df.dropna(subset=["RMSE Score", "Test R2 Score"]).copy()
    g["config_label"] = g[key_cols].astype(str).agg(" | ".join, axis=1)

    top_r2 = g.sort_values("Test R2 Score", ascending=False).head(top_n)
    plt.figure()
    plt.barh(top_r2["config_label"][::-1], top_r2["Test R2 Score"][::-1])
    plt.xlabel("Test R2 Score")
    plt.title(f"Top {top_n} Configurations by R2")
    plt.yticks(fontsize=6)
    savefig(out_dir, f"top_{top_n}_by_r2.png")

    top_rmse = g.sort_values("RMSE Score", ascending=True).head(top_n)
    plt.figure()
    plt.barh(top_rmse["config_label"][::-1], top_rmse["RMSE Score"][::-1])
    plt.xlabel("RMSE Score")
    plt.title(f"Top {top_n} Configurations by Lowest RMSE")
    plt.yticks(fontsize=6)
    savefig(out_dir, f"top_{top_n}_by_rmse.png")


def plot_seed_stability(df, out_dir):
    """Std(RMSE) vs mean(RMSE) over seeds."""
    if "Seed" not in df.columns:
        return

    key_cols = [c for c in [
        "Normalization", "Regressing out", "Points", "LayerGroup",
        "PCA Components", "Regression Type", "Test Split"
    ] if c in df.columns]

    g = df.dropna(subset=["Seed", "RMSE Score"]).copy()
    g["Seed"] = pd.to_numeric(g["Seed"], errors="coerce")

    agg = (
        g.groupby(key_cols)
        .agg(rmse_mean=("RMSE Score", "mean"),
             rmse_std=("RMSE Score", "std"),
             count=("RMSE Score", "size"))
        .reset_index()
    )
    agg = agg[agg["count"] >= 2]
    if len(agg) == 0:
        return

    plt.figure()
    plt.scatter(agg["rmse_mean"], agg["rmse_std"], s=18, alpha=0.7)
    plt.xlabel("Mean RMSE across seeds")
    plt.ylabel("Std RMSE across seeds")
    plt.title("Seed Stability: Std vs Mean RMSE")
    savefig(out_dir, "seed_stability_std_vs_mean_rmse.png")


# -----------------------------
# NEW LayerGroup Accuracy Plotters
# -----------------------------
def plot_layergroup_boxplots(df, out_dir):
    """Direct low vs mid vs high accuracy comparison with boxplots."""
    if "LayerGroup" not in df.columns:
        return

    g = df.dropna(subset=["LayerGroup", "RMSE Score", "Test R2 Score"]).copy()
    order = [x for x in ["low", "mid", "high"] if x in set(g["LayerGroup"].str.lower())]

    def pull(group, col):
        return g[g["LayerGroup"].str.lower() == group][col].values

    # RMSE boxplot
    plt.figure()
    data = [pull(gr, "RMSE Score") for gr in order]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel("RMSE")
    plt.title("RMSE by ProtBERT LayerGroup (low vs mid vs high)")
    savefig(out_dir, "box_rmse_by_layergroup.png")

    # R2 boxplot
    plt.figure()
    data = [pull(gr, "Test R2 Score") for gr in order]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel("Test R2")
    plt.title("R2 by ProtBERT LayerGroup (low vs mid vs high)")
    savefig(out_dir, "box_r2_by_layergroup.png")


def plot_layergroup_means(df, out_dir):
    """Mean ± std bars of RMSE and R2 for low/mid/high."""
    if "LayerGroup" not in df.columns:
        return

    g = df.dropna(subset=["LayerGroup", "RMSE Score", "Test R2 Score"]).copy()
    g["LayerGroup"] = g["LayerGroup"].str.lower()

    agg = g.groupby("LayerGroup").agg(
        rmse_mean=("RMSE Score", "mean"),
        rmse_std=("RMSE Score", "std"),
        r2_mean=("Test R2 Score", "mean"),
        r2_std=("Test R2 Score", "std"),
        n=("RMSE Score", "size")
    ).reset_index()

    # order low->mid->high if present
    order = [x for x in ["low", "mid", "high"] if x in set(agg["LayerGroup"])]
    agg = agg.set_index("LayerGroup").loc[order].reset_index()

    # RMSE bar
    plt.figure()
    plt.bar(agg["LayerGroup"], agg["rmse_mean"], yerr=agg["rmse_std"])
    plt.ylabel("RMSE (mean ± std)")
    plt.xlabel("LayerGroup")
    plt.title("Mean RMSE by ProtBERT LayerGroup")
    savefig(out_dir, "mean_rmse_by_layergroup.png")

    # R2 bar
    plt.figure()
    plt.bar(agg["LayerGroup"], agg["r2_mean"], yerr=agg["r2_std"])
    plt.ylabel("Test R2 (mean ± std)")
    plt.xlabel("LayerGroup")
    plt.title("Mean R2 by ProtBERT LayerGroup")
    savefig(out_dir, "mean_r2_by_layergroup.png")


def plot_rmse_r2_vs_pca_by_layergroup(df, out_dir):
    """
    Line plots of RMSE and R2 vs PCA Components, grouped by LayerGroup.
    This shows whether low/mid/high layers improve differently with PCA depth.
    """
    if "LayerGroup" not in df.columns or "PCA Components" not in df.columns:
        return

    g = df.dropna(subset=["LayerGroup", "PCA Components", "RMSE Score", "Test R2 Score"]).copy()
    g["LayerGroup"] = g["LayerGroup"].str.lower()
    g["PCA Components"] = pd.to_numeric(g["PCA Components"], errors="coerce").astype(int)

    agg = (
        g.groupby(["LayerGroup", "PCA Components"])
        .agg(rmse_mean=("RMSE Score", "mean"),
             rmse_std=("RMSE Score", "std"),
             r2_mean=("Test R2 Score", "mean"),
             r2_std=("Test R2 Score", "std"))
        .reset_index()
        .sort_values("PCA Components")
    )

    order = [x for x in ["low", "mid", "high"] if x in set(agg["LayerGroup"])]

    # RMSE vs PCA by LayerGroup
    plt.figure()
    for lg in order:
        sub = agg[agg["LayerGroup"] == lg]
        plt.plot(sub["PCA Components"], sub["rmse_mean"], marker="o", label=lg)
        plt.fill_between(
            sub["PCA Components"],
            sub["rmse_mean"] - sub["rmse_std"].fillna(0),
            sub["rmse_mean"] + sub["rmse_std"].fillna(0),
            alpha=0.15
        )
    plt.xlabel("PCA Components")
    plt.ylabel("RMSE (mean ± std)")
    plt.title("RMSE vs PCA Components by LayerGroup")
    plt.legend()
    savefig(out_dir, "rmse_vs_pca_by_layergroup.png")

    # R2 vs PCA by LayerGroup
    plt.figure()
    for lg in order:
        sub = agg[agg["LayerGroup"] == lg]
        plt.plot(sub["PCA Components"], sub["r2_mean"], marker="o", label=lg)
        plt.fill_between(
            sub["PCA Components"],
            sub["r2_mean"] - sub["r2_std"].fillna(0),
            sub["r2_mean"] + sub["r2_std"].fillna(0),
            alpha=0.15
        )
    plt.xlabel("PCA Components")
    plt.ylabel("Test R2 (mean ± std)")
    plt.title("R2 vs PCA Components by LayerGroup")
    plt.legend()
    savefig(out_dir, "r2_vs_pca_by_layergroup.png")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".", help="Folder with CSVs")
    ap.add_argument("--out_dir", default="loss_plots", help="Where to save figures")
    ap.add_argument("--top_n", type=int, default=20, help="Top N configs to plot")
    args = ap.parse_args()

    dfs = read_all_csvs(args.data_dir)
    big = concat_loss_tables(dfs)

    big = ensure_numeric(big, ["PCA Components", "Test R2 Score", "RMSE Score", "Combined Score", "Seed"])

    print("[INFO] Combined loss table shape:", big.shape)
    print("[INFO] Columns:", list(big.columns))

    # Core plots
    plot_rmse_r2_by_pca(big, args.out_dir)
    plot_box_rmse_r2_by_regtype(big, args.out_dir)
    plot_r2_vs_rmse_scatter(big, args.out_dir)
    plot_heatmap_mean_rmse(big, args.out_dir, row="LayerGroup", col="PCA Components")
    plot_heatmap_mean_rmse(big, args.out_dir, row="Regression Type", col="PCA Components")
    plot_top_configs(big, args.out_dir, top_n=args.top_n)
    plot_seed_stability(big, args.out_dir)

    # NEW low/mid/high layer accuracy plots
    plot_layergroup_boxplots(big, args.out_dir)
    plot_layergroup_means(big, args.out_dir)
    plot_rmse_r2_vs_pca_by_layergroup(big, args.out_dir)

    # Save merged CSV
    merged_path = os.path.join(args.out_dir, "merged_losses_clean.csv")
    big.to_csv(merged_path, index=False)
    print(f"[SAVED] {merged_path}")


if __name__ == "__main__":
    main()