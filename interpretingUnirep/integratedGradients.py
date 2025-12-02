"""
Integrated Gradients for UniRep + PCA + Linear Regression Rg model.

This script implements Integrated Gradients (Sundararajan, Taly, Yan,
"Axiomatic Attribution for Deep Networks", ICML 2017) for a sequence-to-scalar
model:

    sequence string
      -> UniRep (AAEmbedding + mLSTM)
      -> per-position hidden states h_t
      -> average hidden state h_avg
      -> PCA projection
      -> linear regression
      -> predicted Rg

We compute attributions with respect to the one-hot-encoded sequence input.
"""

import joblib
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.example_libraries import stax

from jax_unirep.layers import AAEmbedding, mLSTM, mLSTMHiddenStates
from jax_unirep.utils import load_params  # or your own loader for evotuned weights


# ============================================================
# 1. Amino-acid alphabet and one-hot encoding
# ============================================================

# From jax_unirep
aa_to_int = {
    "-": 0,
    "M": 1,
    "R": 2,
    "H": 3,
    "K": 4,
    "D": 5,
    "E": 6,
    "S": 7,
    "T": 8,
    "N": 9,
    "Q": 10,
    "C": 11,
    "U": 12,
    "G": 13,
    "P": 14,
    "A": 15,
    "V": 16,
    "I": 17,
    "F": 18,
    "Y": 19,
    "W": 20,
    "L": 21,
    "O": 22,  # Pyrrolysine
    "X": 23,  # Unknown
    "Z": 23,  # Glu/Gln ambiguity
    "B": 23,  # Asn/Asp ambiguity
    "J": 23,  # Leu/Ile ambiguity
    "start": 24,
    "stop": 25,
}

proposal_valid_letters = "ACDEFGHIKLMNPQRSTVWY"

VOCAB_SIZE = 26
UNKNOWN_INDEX = aa_to_int["X"]  # 23

def one_hot_encode_sequence(seq: str) -> jnp.ndarray:
    """
    One-hot encode a protein sequence using the UniRep aa_to_int mapping.

    Parameters
    ----------
    seq : str
        Amino-acid sequence (e.g., 'MKTLLILAVALAVFAA').
        Characters should be in the UniRep vocabulary:
        '-', A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,
        plus rare/ambiguous ones (O, U, B, Z, J, X).

    Returns
    -------
    X : jnp.ndarray, shape (L, 26)
        One-hot encoding of the sequence, where L = len(seq) and
        the last dimension matches the UniRep vocabulary size.
        Unknown characters are mapped to the 'X' class (index 23).
    """
    seq = seq.strip()
    L = len(seq)
    X_np = np.zeros((L, VOCAB_SIZE), dtype=np.float32)

    for t, aa in enumerate(seq):
        # Normal AAs are single characters; 'start'/'stop' tokens are not expected
        # to appear in raw sequences you pass to get_reps.
        idx = aa_to_int.get(aa, UNKNOWN_INDEX)
        X_np[t, idx] = 1.0

    return jnp.array(X_np)

# ============================================================
# 2. UniRep model (AAEmbedding + mLSTM + mLSTMHiddenStates)
# ============================================================

# Build a UniRep model that outputs per-position hidden states.
# This mirrors the architecture used internally by get_reps.
unirep_init, unirep_apply = stax.serial(
    AAEmbedding(10),         # amino-acid embedding dim (10 is UniRep default)
    mLSTM(1900),             # multiplicative LSTM with 1900 hidden units
    mLSTMHiddenStates(),     # output shape: (batch, T, 1900)
)

_, unirep_params = unirep_init(jax.random.PRNGKey(0), input_shape=(-1, VOCAB_SIZE))

# Load UniRep parameters compatible with this architecture.
# If you have evotuned weights, load those instead of paper_weights.
unirep_params = load_params(paper_weights=1900)


# ============================================================
# 3. PCA + regression parameters (from sklearn)
# ============================================================

# You need to fit PCA and regression externally (using sklearn or similar)
# on your training pipeline: h_avg -> PCA -> regression.
#
# Here we assume you have:
#   - pca : fitted sklearn.decomposition.PCA
#   - reg : fitted sklearn.linear_model (or similar)
#
# Example (UNCOMMENT and adapt in your actual code):
#
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
#
# pca = ...  # fitted PCA on h_avg
# reg = ...  # fitted regression on PCA features


# Placeholder: supply real sklearn objects here.
pca = joblib.load("unirep_pca.joblib")
reg = joblib.load("unirep_krr.joblib") 

if (pca is not None) and (reg is not None):
    # Extract PCA and regression parameters
    C_np = pca.components_.astype(np.float32)   # shape (D_pca, 1900)
    mu_np = pca.mean_.astype(np.float32)        # shape (1900,)
    
    # ---------------------------------------------
    # Extract KernelRidge (RBF) parameters from GridSearchCV
    # ---------------------------------------------

    # reg: your GridSearchCV object (or a plain KernelRidge).
    best_reg = getattr(reg, "best_estimator_", reg)

    # Sanity: make sure it's KernelRidge with rbf kernel
    if getattr(best_reg, "kernel", None) != "rbf":
        raise ValueError(f"Expected KernelRidge with kernel='rbf', got {best_reg}")

    # Training PCA features used by KernelRidge
    X_fit_np = best_reg.X_fit_.astype(np.float32)           # (n_train, D_pca)

    # Dual coefficients (alpha_i in the dual representation)
    dual_np = np.asarray(best_reg.dual_coef_, dtype=np.float32).ravel()  # (n_train,)

    # Gamma for the RBF kernel: k(z, z') = exp(-gamma * ||z - z'||^2)
    gamma_value = best_reg.gamma
    if not np.isscalar(gamma_value):
        raise ValueError(
            f"KernelRidge.gamma should be numeric after fit; got {gamma_value!r}"
        )

    # KernelRidge typically has no intercept, but we allow for intercept_ if set
    intercept_value = getattr(best_reg, "intercept_", 0.0)

    # Convert to JAX arrays
    Z_fit = jnp.array(X_fit_np)                         # (n_train, D_pca)
    dual = jnp.array(dual_np)                           # (n_train,)
    gamma = jnp.array(float(gamma_value), dtype=jnp.float32)
    krr_intercept = jnp.array(float(intercept_value), dtype=jnp.float32)

    C = jnp.array(C_np)
    mu = jnp.array(mu_np)
else:
    # If you haven't plugged PCA/reg yet, define dummy values to avoid NameErrors.
    # Replace these with real ones before using integrated_gradients_seq.
    C = None
    mu = None
    Z_fit = None
    dual = None
    gamma = None
    krr_intercept = None


# ============================================================
# 4. Prediction function: one-hot -> UniRep -> PCA -> regression -> Rg
# ============================================================

def predict_rg_from_onehot_with_pca_krr(
    X_LA: jnp.ndarray,
    unirep_params,
    C: jnp.ndarray,
    mu: jnp.ndarray,
    Z_fit: jnp.ndarray,
    dual: jnp.ndarray,
    gamma: jnp.ndarray,
    krr_intercept: jnp.ndarray,
) -> jnp.ndarray:
    """
    Scalar Rg prediction from a one-hot-encoded sequence using:

        UniRep -> average hidden state -> PCA -> KernelRidge (rbf).

    Parameters
    ----------
    X_LA : jnp.ndarray, shape (L, 26)
        One-hot encoding of a single sequence.
    unirep_params :
        Parameters for UniRep model (AAEmbedding + mLSTM + mLSTMHiddenStates).
    C : jnp.ndarray, shape (D_pca, 1900)
        PCA components (rows = principal axes).
    mu : jnp.ndarray, shape (1900,)
        PCA mean vector in UniRep space.
    Z_fit : jnp.ndarray, shape (n_train, D_pca)
        Training PCA features used by KernelRidge (best_reg.X_fit_).
    dual : jnp.ndarray, shape (n_train,)
        Dual coefficients (best_reg.dual_coef_).
    gamma : jnp.ndarray, scalar
        RBF kernel gamma parameter: k(z,z') = exp(-gamma * ||z - z'||^2).
    krr_intercept : jnp.ndarray, scalar
        Intercept term for KernelRidge (usually 0, but included for completeness).

    Returns
    -------
    y_pred : jnp.ndarray, scalar
        Predicted Rg.
    """
    # ---- UniRep forward pass ----
    # X_LA is already shape (L, 26); this matches input_shape=(-1, 26)
    # used when we called unirep_init(..., input_shape=(-1, 26)).
    h_states = unirep_apply(unirep_params, X_LA)  # (L, 1900)

    # Average hidden state over positions (no padding assumed).
    h_avg = h_states.mean(axis=0)                 # (1900,)

    # ---- PCA transform ----
    # z = (h_avg - mu) @ C.T        -> (D_pca,)
    z = (h_avg - mu) @ C.T                         # (D_pca,)

    # ---- KernelRidge with RBF kernel ----
    # Compute RBF kernel between z and each training point z_i in Z_fit
    # Z_fit: (n_train, D_pca), z: (D_pca,)
    diff = Z_fit - z                               # (n_train, D_pca)
    sq_dists = jnp.sum(diff * diff, axis=1)        # (n_train,)

    k = jnp.exp(-gamma * sq_dists)                 # (n_train,)

    # KernelRidge prediction: sum_i dual_i * k(z, z_i) + intercept
    y_pred = jnp.dot(dual, k) + krr_intercept      # scalar

    return y_pred



# ============================================================
# 5. Integrated Gradients for a single sequence
# ============================================================

def integrated_gradients_seq(
    seq: str,
    unirep_params,
    C: jnp.ndarray,
    mu: jnp.ndarray,
    Z_fit: jnp.ndarray,
    dual: jnp.ndarray,
    gamma: jnp.ndarray,
    krr_intercept: jnp.ndarray,
    baseline_type: str = "polyA",
    m: int = 50,
):
    """
    Compute Integrated Gradients (IG) attributions for a single protein sequence
    under the model:

        seq -> UniRep -> average hidden state -> PCA -> KernelRidge (rbf) -> Rg.

    Parameters
    ----------
    seq : str
        Protein sequence of length L.
    unirep_params :
        UniRep parameters.
    C : jnp.ndarray, shape (D_pca, 1900)
        PCA components.
    mu : jnp.ndarray, shape (1900,)
        PCA mean.
    Z_fit : jnp.ndarray, shape (n_train, D_pca)
        Training PCA features for KernelRidge.
    dual : jnp.ndarray, shape (n_train,)
        Dual coefficients of KernelRidge.
    gamma : jnp.ndarray, scalar
        RBF gamma.
    krr_intercept : jnp.ndarray, scalar
        KernelRidge intercept (usually 0).
    baseline_type : {"polyA", "zero"}
        Baseline for IG.
    m : int
        Number of IG steps in the Riemann approximation.

    Returns
    -------
    ig_full : jnp.ndarray, shape (L, 26)
        IG attributions per position and channel.
    ig_pos : jnp.ndarray, shape (L,)
        IG attributions per residue (sum across 26 channels).
    ig_chan : jnp.ndarray, shape (26,)
        IG attributions per channel, summed across positions.
    """
    # 1) Encode input sequence (L, 26)
    X = one_hot_encode_sequence(seq)
    L, A = X.shape

    # 2) Build baseline X0
    if baseline_type == "polyA":
        baseline_seq = "A" * L
        X0 = one_hot_encode_sequence(baseline_seq)  # (L, 26)
    elif baseline_type == "zero":
        X0 = jnp.zeros_like(X)
    elif baseline_type == "polyG":
        baseline_seq = "G" * L
        X0 = one_hot_encode_sequence(baseline_seq)  # (L, 26)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    # 3) Define scalar model F(X) using KernelRidge-based predictor
    def F(X_in):
        return predict_rg_from_onehot_with_pca_krr(
            X_in,
            unirep_params,
            C,
            mu,
            Z_fit,
            dual,
            gamma,
            krr_intercept,
        )

    # 4) Path alphas for Riemann sum
    alphas = jnp.linspace(0.0, 1.0, m)  # (m,)

    def interpolate(alpha):
        return X0 + alpha * (X - X0)     # (L, 26)

    # 5) Gradient of F w.r.t. input X
    grad_F_wrt_X = jax.grad(F)          # X_in -> (L, 26)

    def grad_at_alpha(alpha):
        X_alpha = interpolate(alpha)
        return grad_F_wrt_X(X_alpha)    # (L, 26)

    # 6) Evaluate gradients along the path: (m, L, 26)
    grads = jax.vmap(grad_at_alpha)(alphas)

    # 7) Average gradients along the path: (L, 26)
    avg_grads = grads.mean(axis=0)

    # 8) Integrated Gradients: (X - X0) * average_gradient
    ig_full = (X - X0) * avg_grads      # (L, 26)

    # 9) Aggregate:
    ig_pos = ig_full.sum(axis=1)        # (L,)    per-position
    ig_chan = ig_full.sum(axis=0)       # (26,)   per-channel

    return ig_full, ig_pos, ig_chan



# ============================================================
# 6. Example usage (once PCA and reg are set)
# ============================================================

# if __name__ == "__main__":
#     # You MUST replace pca and reg above with your real fitted models,
#     # then rebuild C, mu, w_reg, b_reg.

#     if (C is None) or (mu is None) or (Z_fit is None) or (dual is None) or (gamma is None) or (krr_intercept is None):
#         raise RuntimeError(
#             "Please plug your fitted PCA and regression objects into this script "
#             "and rebuild C, mu, w_reg, b_reg before running IG."
#         )

#     sequence = "MKTLLILAVALAVFAA"  # example sequence

#     ig_full, ig_pos, ig_chan = integrated_gradients_seq(
#         sequence,
#         unirep_params,
#         C,
#         mu,
#         Z_fit,
#         dual,
#         gamma,
#         krr_intercept,
#         baseline_type="polyA",
#         m=50,
#     )

#     print("Sequence:", sequence)
#     print("Sum of IG attributions:", float(ig_full.sum()))
#     print("Per-residue IG attributions:", np.array(ig_pos))
#     print("Per-channel IG attributions:", [a + ": " + str(val) for a, val in zip(aa_to_int, np.array(ig_chan))])
    

if __name__ == "__main__":
    index_to_label = [""] * VOCAB_SIZE
    for aa, idx in aa_to_int.items():
        if idx < VOCAB_SIZE and index_to_label[idx] == "":
            index_to_label[idx] = aa

    # ---------------------------------------------
    # Global accumulators
    # ---------------------------------------------
    global_aa_scores = np.zeros(VOCAB_SIZE, dtype=np.float64)

    max_k = 11
    # global_kmer_scores[k][kmer_string] = cumulative IG score
    global_kmer_scores = {k: {} for k in range(2, max_k + 1)}

    # ---------------------------------------------
    # Load sequences
    # ---------------------------------------------
    df = pd.read_csv("training/all_points.csv")

    if "Sequence" not in df.columns:
        raise KeyError("Expected a 'Sequence' column in all_points.csv")

    sequences = df["Sequence"].astype(str).tolist()

    # ---------------------------------------------
    # Main loop over all sequences
    # ---------------------------------------------
    for idx_seq, seq in enumerate(sequences):
        seq = seq.strip()
        if not seq:
            continue  # skip empty sequences

        # Compute IG for this sequence (polyA baseline)
        ig_full, ig_pos, ig_chan = integrated_gradients_seq(
            seq,
            unirep_params,
            C,
            mu,
            Z_fit,
            dual,
            gamma,
            krr_intercept,
            baseline_type="polyG",
            m=20,  # or whatever you chose
        )

        # Convert JAX arrays to numpy for accumulation
        ig_full_np = np.array(ig_full, dtype=np.float32)  # (L, 26)
        ig_pos_np = np.array(ig_pos, dtype=np.float32)    # (L,)

        # 1) Per-AA-type scores for this sequence
        #    Sum over positions -> (26,)
        ig_chan_np = ig_full_np.sum(axis=0)
        global_aa_scores += ig_chan_np

        # 2) k-mer scores (k = 2...11)
        L = len(seq)
        for k in range(2, max_k + 1):
            if L < k:
                continue
            d_k = global_kmer_scores[k]
            # Slide a window of length k
            for i in range(L - k + 1):
                kmer = seq[i : i + k]       # substring
                # Sum of per-residue IG_pos over the window
                kmer_score = ig_pos_np[i : i + k].sum()
                d_k[kmer] = d_k.get(kmer, 0.0) + kmer_score

        # (Optional) progress print
        if (idx_seq + 1) % 10 == 0:
            print(f"Processed {idx_seq + 1} sequences")

    # ---------------------------------------------
    # Results:
    #   - global_aa_scores: per-channel IG sums (len 26)
    #   - index_to_label:   labels for each channel index
    #   - global_kmer_scores[k]: dict of k-mer -> score
    # ---------------------------------------------

    # ---------------------------------------------
    # Save per–amino-acid IG scores
    # ---------------------------------------------
    aa_df = pd.DataFrame({
        "channel_index": np.arange(VOCAB_SIZE),
        "aa_label": index_to_label,
        "ig_score": global_aa_scores,
    })
    aa_df.to_csv("aa_ig_scores.csv", index=False)

    # ---------------------------------------------
    # Save k-mer IG scores for all k = 2..11
    # ---------------------------------------------
    kmer_rows = []
    for k, d_k in global_kmer_scores.items():
        for kmer, score in d_k.items():
            kmer_rows.append((k, kmer, score))

    kmer_df = pd.DataFrame(kmer_rows, columns=["k", "kmer", "ig_score"])
    kmer_df.to_csv("kmer_ig_scores.csv", index=False)

    # Example: zip AA labels with scores
    aa_scores_labeled = list(zip(index_to_label, global_aa_scores))

    # Sort by score descending (optional)
    aa_scores_labeled_sorted = sorted(
        aa_scores_labeled,
        key=lambda x: x[1],
        reverse=True,
    )


    print("Per-AA IG scores (label, score):")
    for label, score in aa_scores_labeled_sorted:
        print(f"{label:>5s}  {score: .6f}")

    # Example: top k-mers for a given k
    k = 3
    top_n = 20
    print(f"\nTop {top_n} {k}-mers by IG score:")
    for kmer, score in sorted(global_kmer_scores[k].items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"{kmer:>8s}  {score: .6f}")