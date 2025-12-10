# unirep_sliding_window.py

import numpy as np
from jax_unirep import get_reps
import csv
import pickle
import joblib

def unirep_predict_rg(seq, regressor, pca=None):
    """
    Predict Rg for a protein sequence using UniRep + your top model.

    Parameters
    ----------
    seq : str
        Amino-acid sequence.
    regressor : sklearn-like model
        Must support .predict(X) where X shape is (1, D).
    pca : sklearn.decomposition.PCA or None
        If you used PCA when training the regressor, pass that here.

    Returns
    -------
    float
        Predicted Rg.
    """
    # get_reps can take a single string or a list; we assume single here.
    # It returns (h_avg, h_final, c_final), each shape (1, 1900).
    h_avg, _, _ = get_reps(seq)  # shape (1, 1900)

    X = h_avg
    if pca is not None:
        X = pca.transform(X)

    y_pred = regressor.predict(X)[0]
    return float(y_pred)


def sliding_window_unirep(
    seq,
    regressor,
    window=5,
    pca=None,
    mode="delete",
):
    """
    Naive sliding-window occlusion for UniRep:
    for each window [start:end), actually modify the sequence,
    recompute UniRep embedding from scratch, and see ΔRg.

    Parameters
    ----------
    seq : str
        Original amino-acid sequence.
    regressor : sklearn-like model
        Trained Rg model on UniRep h_avg (optionally PCA-reduced).
    window : int, default=5
        Window size in residues.
    pca : PCA or None
        PCA used in training, if any.
    mode : {'delete', 'mask'}, default='delete'
        - 'delete': remove the window residues from the sequence.
        - 'mask'  : replace residues in the window with 'X'.

    Returns
    -------
    deltas : np.ndarray, shape (L - window + 1,)
        ΔRg for each window (rg_occ - baseline).
    baseline : float
        Rg prediction for the full, unmodified sequence.
    fragments : list[str]
        The sequence fragments that were occluded.
    indices : list[tuple[int, int]]
        (start, end) indices of each window, 0-based, end-exclusive.
    """
    L = len(seq)
    if window > L:
        raise ValueError(f"window ({window}) > sequence length ({L})")

    # 1. Baseline prediction on full sequence
    baseline = unirep_predict_rg(seq, regressor, pca=pca)

    deltas = []
    fragments = []
    indices = []

    # 2. Slide the window
    for start in range(L - window + 1):
        if start % 20 == 0:
            print(f"  Processing window starting at residue {start}")
        end = start + window
        frag = seq[start:end]
        fragments.append(frag)
        indices.append((start, end))

        if mode == "delete":
            # Remove these residues entirely
            seq_occ = seq[:start] + seq[end:]
        elif mode == "mask":
            # Replace them with 'X' (or choose another neutral AA)
            seq_occ = seq[:start] + ("X" * window) + seq[end:]
        else:
            raise ValueError("mode must be 'delete' or 'mask'")

        # If deletion leaves an empty sequence, skip (cannot compute UniRep)
        if len(seq_occ) == 0:
            # You can choose to append np.nan here instead
            deltas.append(np.nan)
            continue

        rg_occ = unirep_predict_rg(seq_occ, regressor, pca=pca)
        delta = rg_occ - baseline
        deltas.append(delta)

    return np.array(deltas, dtype=np.float32), baseline, fragments, indices



# Load model and PCA once
regr_model = joblib.load("interpretingUnirep/unirep_krr.joblib")
pca = joblib.load("interpretingUnirep/unirep_pca.joblib")

# Output CSV: one row per (window_size, sequence, window_position)
out_path = "unirep_sliding_windows_seqDelete.csv"

with open(out_path, "w", newline="") as out_f:
    writer = csv.writer(out_f)
    # Header row: tweak columns as you like
    writer.writerow([
        "window_size",
        "sequence_index",
        "window_index",
        "start",
        "end",
        "baseline_rg",
        "delta_rg",
        "fragment",
        "sequence_id"  # optional: e.g. original row index or any id
    ])

    # Loop over window sizes
    for w in range(1, 11):
        print(f"Window size {w}")

        # Re-open the input file for each window size to avoid keeping all sequences in RAM
        with open("training/all_points.csv", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            for seq_idx, row in enumerate(reader):
                seq = row[0]  # assuming sequence is in column 0

                # Run sliding-window occlusion for this sequence and window size
                deltas, baseline, frags, idxs = sliding_window_unirep(
                    seq,
                    regr_model,
                    window=w,
                    pca=pca,
                    mode="delete",
                )

                # Stream each window result as its own CSV row
                for win_idx, (delta, frag, (start, end)) in enumerate(zip(deltas, frags, idxs)):
                    writer.writerow([
                        w,                 # window_size
                        seq_idx,           # sequence_index within the CSV (0-based after header)
                        win_idx,           # window_index along this sequence
                        int(start),
                        int(end),
                        float(baseline),
                        float(delta),
                        frag,
                        seq_idx            # sequence_id (here just same as seq_idx; replace if you have real IDs)
                    ])

                # Let Python reclaim these per-sequence arrays quickly
                del deltas, baseline, frags, idxs

                if (seq_idx + 1) % 1 == 0:
                    print(f"  {seq_idx + 1} sequences processed for window size {w}")

print(f"Done. Results written to {out_path}")
