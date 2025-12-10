import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_window_omissions(csv_path):
    """
    Load rows from window_omissions_all.csv

    Assumes header with columns: omitted_seq, delta_rg, start_pos
    """
    frags = []
    deltas = []
    starts = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frag = row["omitted_seq"]
            d_rg = float(row["delta_rg"])
            start = int(row["start_pos"])
            frags.append(frag)
            deltas.append(d_rg)
            starts.append(start)

    return np.array(frags, dtype=object), np.array(deltas, dtype=float), np.array(starts, dtype=int)


def aggregate_by_motif(frags, deltas, k=None, agg="mean"):
    """
    Group by exact motif string (optionally with fixed length k) and
    aggregate delta_rg.

    Parameters
    ----------
    frags : array-like of str
    deltas : array-like of float
    k : int or None
        If not None, only keep motifs of length k.
    agg : {'mean', 'median'}
        How to aggregate multiple occurrences of the same motif.

    Returns
    -------
    motifs : list of str
    scores : list of float
    counts : list of int
    """
    stats = defaultdict(list)
    for frag, d in zip(frags, deltas):
        if k is not None and len(frag) != k:
            continue
        stats[frag].append(d)

    motifs = []
    scores = []
    counts = []
    for frag, vals in stats.items():
        arr = np.array(vals, dtype=float)
        if agg == "median":
            score = float(np.median(arr))
        else:
            score = float(arr.mean())
        motifs.append(frag)
        scores.append(score)
        counts.append(len(arr))

    return motifs, scores, counts


def plot_top_compacting_motifs_from_csv(
    csv_path,
    k,
    top_n=20,
    min_count=5,
    only_negative=True,
):
    """
    Read window_omissions_all.csv and plot the top-N *compacting* motifs
    of length k as a horizontal bar plot, very similar to plot_top_motifs.

    Parameters
    ----------
    csv_path : str
        Path to window_omissions_all.csv
    k : int
        Motif length (len(omitted_seq)).
    top_n : int
        How many motifs to display.
    min_count : int
        Require at least this many occurrences of a motif.
    only_negative : bool
        If True, only consider motifs whose aggregated delta_rg < 0
        (they reduce Rg).
    """
    frags, deltas, starts = load_window_omissions(csv_path)

    # aggregate by motif
    motifs, scores, counts = aggregate_by_motif(frags, deltas, k=k, agg="mean")

    motifs = np.array(motifs, dtype=object)
    scores = np.array(scores, dtype=float)
    counts = np.array(counts, dtype=int)

    # filter by count and sign
    mask = counts >= min_count
    if only_negative:
        mask &= (scores < 0)

    motifs = motifs[mask]
    scores = scores[mask]
    counts = counts[mask]

    if len(motifs) == 0:
        print("No motifs passed the filters.")
        return

    # sort by "most compacting" (most negative)
    order = np.argsort(scores)  # ascending; most negative first
    motifs = motifs[order][:top_n]
    scores = scores[order][:top_n]
    counts = counts[order][:top_n]

    # plot like plot_top_motifs(compacting=..., k=k)
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(motifs))

    plt.barh(motifs, scores, color="blue")
    plt.gca().invert_yaxis()

    for i, (s, c) in enumerate(zip(scores, counts)):
        plt.text(s, i, f"  n={c}", va="center")

    plt.xlabel("Mean ΔRg (occluded - baseline)")
    plt.title(f"Top compacting motifs (k={k}) from window_omissions_all")
    plt.tight_layout()
    plt.show()


plot_top_compacting_motifs_from_csv(
    "interpretingUnirep/window_omissions_all.csv",
    k=3,
    top_n=30,
    min_count=5,
    only_negative=True,
)
