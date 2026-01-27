import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from collections import defaultdict

# ---------------------------------------------------------
# 1. Load window omissions from CSV
# ---------------------------------------------------------

def load_windows(csv_path, k=None):
    """
    Load (fragment, delta_rg) pairs from window_omissions_all.csv.
    Assumes header: omitted_seq, delta_rg, start_pos
    """
    frags = []
    dRg   = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frag = row["omitted_seq"].strip()
            if not frag: continue
            
            drg  = float(row["delta_rg"])
            if (k is not None) and (len(frag) != k):
                continue
            frags.append(frag)
            dRg.append(drg)

    return np.array(frags, dtype=object), np.array(dRg, dtype=float)


# ---------------------------------------------------------
# 2. Normalization and motif-weight aggregation
# ---------------------------------------------------------

def zscoreNormalize(dRg_seq, eps=1e-12, clip=None):
    dRg = np.asarray(dRg_seq, dtype=float)
    mu  = dRg.mean()
    sigma = dRg.std()

    if sigma < eps:
        z = np.zeros_like(dRg)
    else:
        z = (dRg - mu) / sigma

    if clip is not None:
        z = np.clip(z, -clip, clip)

    w = -z         # sign flip as in original code
    return w


def computeWeightedResidueEffects_from_csv(
    csv_path,
    k,
    min_count=20,
    use_abs_weight=False,
    clip_z=None,
):
    frags, dRg = load_windows(csv_path, k=k)

    if len(frags) == 0:
        print(f"Warning: No fragments of length k={k} found.")
        return []

    dRg_norm = zscoreNormalize(dRg, clip=clip_z)

    sum_effect = [defaultdict(float) for _ in range(k)]
    count      = [defaultdict(float) for _ in range(k)]

    for drg, frag in zip(dRg_norm, frags):
        if abs(drg) < 1e-6:
            continue
        for pos, aa in enumerate(frag):
            weight = abs(drg) if use_abs_weight else 1.0
            sum_effect[pos][aa] += drg * weight
            count[pos][aa]      += weight

    mean_effect = []
    for pos in range(k):
        pos_dict = {}
        for aa, tot_w in sum_effect[pos].items():
            if count[pos][aa] < min_count:
                pos_dict[aa] = 0.0
            else:
                pos_dict[aa] = tot_w / count[pos][aa]
        mean_effect.append(pos_dict)

    return mean_effect


def effects_to_df(mean_effect):
    if not mean_effect:
        return pd.DataFrame()
        
    aas_set = set()
    for pos_dict in mean_effect:
        aas_set.update(pos_dict.keys())
    aas = sorted(aas_set)

    data = []
    for pos_dict in mean_effect:
        row = [pos_dict.get(aa, 0.0) for aa in aas]
        data.append(row)

    df = pd.DataFrame(data, columns=aas)
    # Ensure 1-based indexing for the plot
    df.index = np.arange(1, len(mean_effect) + 1)
    return df


# ---------------------------------------------------------
# 3. Plotting Logic (Grid Layout)
# ---------------------------------------------------------

def plot_grid_logos(csv_path, sizes=[3, 6, 10], min_count=20, clip_z=3.0):
    """
    Creates a grid of logos matching the reference image.
    Ensures column widths are uniform across different sizes.
    Uses 'chemistry' color scheme.
    """
    
    # 1. Determine the maximum size to standardize the X-axis width
    max_k = max(sizes)
    
    # Create the grid
    fig, axes = plt.subplots(nrows=len(sizes), ncols=2, figsize=(10, 8))
    
    for i, k in enumerate(sizes):
        print(f"Processing size k={k}...")
        
        # --- Data Prep ---
        mean_effect = computeWeightedResidueEffects_from_csv(
            csv_path, k=k, min_count=min_count, clip_z=clip_z
        )
        df = effects_to_df(mean_effect)
        
        if not df.empty:
            df_expand  = df.clip(lower=0.0)
            df_compact = (-df).clip(lower=0.0)
        else:
            df_expand = df
            df_compact = df

        # --- Plotting ---
        ax_exp = axes[i, 0]
        ax_comp = axes[i, 1]

        # Plot Expansion (Restore color_scheme='chemistry')
        if not df_expand.empty:
            logomaker.Logo(df_expand, ax=ax_exp, color_scheme='chemistry')
        
        # Plot Compaction (Restore color_scheme='chemistry')
        if not df_compact.empty:
            logomaker.Logo(df_compact, ax=ax_comp, color_scheme='chemistry')

        # --- Formatting for Uniform Widths & Ticks ---
        
        for ax in [ax_exp, ax_comp]:
            # 1. LOCK THE X-LIMITS: 
            # Force every plot to have the same "physical" width per unit 
            ax.set_xlim([0.5, max_k + 0.5])
            
            # 2. ENABLE TICKS FOR EXISTING COLUMNS:
            # Force ticks to appear for positions 1 to k
            ax.set_xticks(range(1, k + 1))
            
            # 3. CONDITIONAL LABELS:
            # Hide labels for non-bottom rows
            if i < len(sizes) - 1:
                ax.set_xticklabels([])
            
            # Remove spines for clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Row Labels (Y-axis label on the far left only)
        ax_exp.set_ylabel(f"Size {k}", fontsize=12, labelpad=10)
        
        # Remove Y-axis ticks
        ax_exp.set_yticks([])
        ax_comp.set_yticks([])
        ax_comp.set_ylabel("") 

        # Column Titles (Top row only)
        if i == 0:
            ax_exp.set_title("Expansion", fontsize=14, pad=10)
            ax_comp.set_title("Compaction", fontsize=14, pad=10)
        else:
            ax_exp.set_title("")
            ax_comp.set_title("")

        # X-axis Labels (Bottom row only)
        if i == len(sizes) - 1:
            ax_exp.set_xlabel("Position in motif", fontsize=12)
            ax_comp.set_xlabel("Position in motif", fontsize=12)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    csv_path = "interpretingUnirep/window_omissions_all.csv" 
    
    # Define sizes
    sizes_to_plot = [3, 6, 10]
    
    plot_grid_logos(
        csv_path,
        sizes=sizes_to_plot,
        min_count=50,   
        clip_z=3.0      
    )