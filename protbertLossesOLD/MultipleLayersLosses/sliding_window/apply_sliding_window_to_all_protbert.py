#!/usr/bin/env python3
# apply_sliding_window_to_all_protbert.py

import argparse, os, json, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------------------------
# Import functions from protbert_sliding_window.py
# -------------------------------------------------------------------

from protbert_sliding_window import (
    embedding_occlusion_effect,
    load_frozen,
    PROTBERT_NAME,
    LAYER_MAP
)

# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--models_root", required=True,
                    help="Folder containing model_1, model_2, ...")

    ap.add_argument("--seq_csv", required=True,
                    help="CSV with a Sequence column containing all sequences.")

    ap.add_argument("--sequence_col", default="Sequence",
                    help="Column name for sequences in seq_csv.")

    ap.add_argument("--out_dir", default="protbert_sliding_results_all_only")

    ap.add_argument("--min_window", type=int, default=1)
    ap.add_argument("--max_window", type=int, default=10)

    ap.add_argument("--occlusion_mode", choices=["zero", "mean"], default="zero")

    ap.add_argument("--device", default="cpu")

    args = ap.parse_args()

    # ---------------------------------------------------------------
    # Load sequences
    # ---------------------------------------------------------------
    df = pd.read_csv(args.seq_csv)
    if args.sequence_col not in df.columns:
        raise KeyError(f"{args.sequence_col} not found in {args.seq_csv}. "
                       f"Columns are: {df.columns.tolist()}")

    seqs = df[args.sequence_col].astype(str).tolist()

    # ---------------------------------------------------------------
    # Load ProtBERT model ONCE (shared across everything)
    # ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(PROTBERT_NAME, do_lower_case=False)
    model = AutoModel.from_pretrained(PROTBERT_NAME).to(args.device)
    model.eval()

    # ---------------------------------------------------------------
    # Prepare output directory
    # ---------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Locate model_1/, model_2/, model_3/, ...
    # ---------------------------------------------------------------
    model_dirs = sorted(
        [os.path.join(args.models_root, d)
         for d in os.listdir(args.models_root)
         if d.startswith("model_")]
    )

    if not model_dirs:
        raise RuntimeError(f"No model_* directories found in {args.models_root}")

    # ---------------------------------------------------------------
    # Run full sliding window attribution
    # ---------------------------------------------------------------
    for mdir in model_dirs:

        # Load this model’s frozen PCA and regressor
        pca, reg, cfg = load_frozen(mdir)
        layer_group = str(cfg["LayerGroup"]).lower()
        if layer_group not in LAYER_MAP:
            raise ValueError(f"Unknown LayerGroup '{layer_group}' in config.json")

        layer_idx = LAYER_MAP[layer_group]

        effects_by_window = []
        windows_by_window = []
        frags_by_window = []
        baselines_all = []

        print(f"\n=== Processing {mdir}  (layer_group={layer_group}, layer={layer_idx}) ===")

        for w in range(args.min_window, args.max_window + 1):

            effects_list = []
            windows_list = []
            frags_list = []
            baselines_list = []

            print(f"  --> Window size {w}")

            for seq in tqdm(seqs, desc=f"{os.path.basename(mdir)} window={w}"):

                baseline, effects, windows, frags = embedding_occlusion_effect(
                    seq,
                    pca, reg,
                    tokenizer, model,
                    layer_idx=layer_idx,
                    window=w,
                    mode=args.occlusion_mode,
                    device=args.device
                )

                baselines_list.append(baseline)
                effects_list.append(effects)
                windows_list.append(windows)
                frags_list.append(frags)

            effects_by_window.append(effects_list)
            windows_by_window.append(windows_list)
            frags_by_window.append(frags_list)
            baselines_all.append(baselines_list)

        # -----------------------------------------------------------
        # Save results for this model
        # -----------------------------------------------------------
        out_mdir = os.path.join(args.out_dir, os.path.basename(mdir))
        os.makedirs(out_mdir, exist_ok=True)

        with open(os.path.join(out_mdir, "effects_by_window.pkl"), "wb") as f:
            pickle.dump(effects_by_window, f)

        with open(os.path.join(out_mdir, "windows_by_window.pkl"), "wb") as f:
            pickle.dump(windows_by_window, f)

        with open(os.path.join(out_mdir, "frags_by_window.pkl"), "wb") as f:
            pickle.dump(frags_by_window, f)

        with open(os.path.join(out_mdir, "baselines.pkl"), "wb") as f:
            pickle.dump(baselines_all, f)

        print(f"[saved] Sliding results -> {out_mdir}")

# -------------------------------------------------------------------

if __name__ == "__main__":
    main()