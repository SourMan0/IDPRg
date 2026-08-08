# FG-Nup case study, leak-free re-run
# -----------------------------------
# For each of four FG-Nups, slide a length-k=3 occlusion window along the
# sequence, ask the leak-free KRR pipeline how the predicted Rg changes, and
# plot the per-residue ΔRg with F (phenylalanine) and G (glycine) highlighted
# -- the two residues that define FG-repeats. Stems are gray for all other
# residues so the FG signal pops.
#
# Differences vs the old version:
#   * Loads the single new artifact krr_pipeline.joblib instead of the
#     separate (leaky) krr.joblib + esm_pca2.joblib pair.
#   * Extracts ESM-6 layer 1 (was layer 3) to match the new best backbone.
#   * Uses pipeline.predict() in one shot instead of a manual pca.transform
#     followed by model.predict().
#   * Saves a single 4-panel figure (PDF + PNG) rather than four separate
#     interactive plt.show() windows.
#   * Drops the duplicate plt.bar call that was drawing each set of bars
#     twice.

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


ESM_LAYER = 1   # matches the new ESM-6 L1 PCA=100 KRR pipeline
WINDOW_K = 3    # original case-study window size

tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
esm.eval()

pipeline = joblib.load('krr_pipeline.joblib')


def get_layer_embeddings(seq):
    """Per-residue embeddings from ESM_LAYER, shape (L, 320), BOS/EOS stripped."""
    tokens = tok(seq, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = esm(**tokens, output_hidden_states=True)
    emb = out.hidden_states[ESM_LAYER][0]
    emb = emb[1:-1]
    return emb


def predict_rg(E):
    """Mean-pool E (L, D) → pipeline.predict on the (1, D) pooled vector."""
    pooled = E.mean(dim=0).cpu().numpy()
    return float(pipeline.predict(pooled.reshape(1, -1))[0])


def occlude(E, start, k, method="zero"):
    E_occ = E.clone()
    if method == "zero":
        E_occ[start:start + k] = 0.0
    elif method == "mean":
        E_occ[start:start + k] = E.mean(dim=0, keepdim=True)
    else:
        raise ValueError(method)
    return E_occ


def sliding_occlusion(seq, k, method="zero"):
    E = get_layer_embeddings(seq)
    rg_ref = predict_rg(E)
    drg = np.zeros(len(seq) - k + 1)
    for start in range(len(seq) - k + 1):
        drg[start] = predict_rg(occlude(E, start, k, method)) - rg_ref
    return drg, rg_ref


# FG-Nup sequences (preserved verbatim from the previous case study)
nups = [
    ("Nup98",
     "GCFNKSFGTPFGGGTGGFGTTSTFGQNTGFGTTSGGAFGTSAFGSSNNTGGLFGNSQTKPGGLFGTSSFSQPATSTSTGFGFGTSTGTANTLFGTASTGTSLFSSQNNAFAQNKPTGFGNFGTSTSSGGLFGTTNTTSNPFGSTSGSLFGPUA"),
    ("Nup49",
     "GCQTSRGLFGNNNTNNINNSSSGMNNASAGLFGSKPUA"),
    ("Nup153 NUS domain",
     "GCPSASPAFGANQTPTFGQSQGASQPNPPGFGSISSSTALFPTGSQPAPPTFGTVSSSSQPPVFGQQPSQSAFGSGTTPNUA"),
    ("Nup153 NUL domain",
     "GCGFKGFDTSSSSSNSAASSSFKFGVSSSSSGPSQTLTSTGNFKFGDQGGFKIGVSSDSGSINPMSEGFKFSKPIGDFKFGVSSESKPEEVKKDSKNDNFKFGLSSGLSNPVUA"),
]

# Highlight F and G (the FG-repeat defining residues); everything else gray.
def bar_colors(seq):
    out = []
    for aa in seq:
        if aa == 'F':
            out.append('dodgerblue')
        elif aa == 'G':
            out.append('forestgreen')
        else:
            out.append('gray')   # medium gray, matches the reference figure
    return out


mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
})

print(f"Pipeline: krr_pipeline.joblib  |  ESM-6 layer {ESM_LAYER}  |  k = {WINDOW_K}")


def slugify(name):
    return name.lower().replace(' ', '_').replace('/', '_')


# One figure per Nup, width scaled to sequence length so that the per-residue
# bar density is visually comparable across the four files.
for name, seq in nups:
    drg, rg_ref = sliding_occlusion(seq, WINDOW_K, method="zero")
    print(f"  {name:<22} L={len(seq):3d}  Rg_ref={rg_ref:6.3f}  "
          f"ΔRg range=[{drg.min():+.4f}, {drg.max():+.4f}]")

    # Wider per-residue so each column has room; small gap (width<1)
    # and a thin black outline so individual bars are visible even when
    # neighbours have similar heights.
    fig_w = max(4.0, len(seq) / 11.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))

    positions = np.arange(len(drg))
    ax.bar(positions, drg,
           color=bar_colors(seq[: len(seq) - WINDOW_K + 1]),
           edgecolor='none', width=0.78)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xlim(-0.5, len(drg) - 0.5)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("ΔRg (Å)")
    for s in ax.spines.values():
        s.set_linewidth(1.0)
        s.set_color('black')
        s.set_visible(True)

    base = f"fgnup_{slugify(name)}"
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    -> {base}.pdf (+ .png)")
