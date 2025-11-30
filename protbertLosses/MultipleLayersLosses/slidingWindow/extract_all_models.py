#!/usr/bin/env python3
# extract_all_models.py

import argparse, json
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_json", default="all_models.json")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)

    # keep only ProtBERT + Points == All (case/whitespace robust)
    df_all = df[
        (df["Model"].astype(str).str.contains("protbert", case=False, na=False))
    ].copy()

    if len(df_all) == 0:
        raise ValueError("No ProtBERT models with Points=='All' found in summary.")

    # sort by Combined Score descending (best first)
    df_all = df_all.sort_values("Combined Score", ascending=False).head(args.top_k)

    configs = []
    for _, row in df_all.iterrows():
        configs.append({
            "LayerGroup": str(row["LayerGroup"]).strip().lower(),   # low/mid/high
            "PCA Components": int(row["PCA Components"]),
            "Regression Type": str(row["Regression Type"]).strip(),
            "Seed": int(row.get("Seed", 1)),
            "Normalization": row.get("Normalization", None),
            "Regressing out": row.get("Regressing out", None),
        })

    with open(args.out_json, "w") as f:
        json.dump(configs, f, indent=2)

    print(f"[saved] {args.out_json} with {len(configs)} models:")
    for c in configs:
        print(c)

if __name__ == "__main__":
    main()