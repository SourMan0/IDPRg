import joblib
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.example_libraries import stax
import csv

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
pca = joblib.load("interpretingUnirep/unirep_pca.joblib")
reg = joblib.load("interpretingUnirep/unirep_krr.joblib") 

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




def _krr_predict_from_h_avg_np(
    h_avg_np: np.ndarray,
    C_np: np.ndarray,
    mu_np: np.ndarray,
    Z_fit_np: np.ndarray,
    dual_np: np.ndarray,
    gamma_val: float,
    intercept_val: float,
) -> float:
    """
    Helper: given a mean hidden state h_avg (1900-dim), compute Rg using:
        h_avg -> PCA -> KernelRidge (rbf).
    Works entirely in NumPy (no gradients needed).
    """
    # PCA: z = (h_avg - mu) @ C.T
    z = (h_avg_np - mu_np) @ C_np.T             # (D_pca,)

    # RBF kernel between z and each training point in Z_fit
    diff = Z_fit_np - z                         # (n_train, D_pca)
    sq_dists = np.sum(diff * diff, axis=1)      # (n_train,)
    k = np.exp(-gamma_val * sq_dists)           # (n_train,)

    # KernelRidge prediction
    y_pred = float(dual_np @ k + intercept_val)
    return y_pred


def window_omission_effects(
    seq: str,
    unirep_params,
    C,
    mu,
    Z_fit,
    dual,
    gamma,
    krr_intercept,
    max_w: int = 11,
):
    """
    For a given sequence, compute the effect of omitting each possible
    contiguous window of hidden states (length 1..max_w) from the mean
    UniRep embedding used for Rg prediction.

    Process for one sequence:
      1) Compute hidden states h_t (L, 1900).
      2) Compute original Rg using mean over all t.
      3) For each window size w = 1..max_w, and each start index n:
           - Exclude steps [n, n+w-1] from the mean,
           - Recompute mean embedding over remaining steps,
           - Predict Rg_window,
           - delta_Rg = Rg_window - Rg_original,
           - Store omitted subsequence seq[n:n+w], delta_Rg, and n (1-based).

    Parameters
    ----------
    seq : str
        Protein sequence of length L.
    unirep_params :
        Parameters for UniRep model (AAEmbedding + mLSTM + mLSTMHiddenStates).
    C : jnp.ndarray, shape (D_pca, 1900)
        PCA components.
    mu : jnp.ndarray, shape (1900,)
        PCA mean.
    Z_fit : jnp.ndarray, shape (n_train, D_pca)
        Training PCA features for KernelRidge.
    dual : jnp.ndarray, shape (n_train,)
        Dual coefficients for KernelRidge.
    gamma : jnp.ndarray, scalar
        RBF gamma parameter.
    krr_intercept : jnp.ndarray, scalar
        KernelRidge intercept (usually 0).
    max_w : int, optional
        Maximum window size to test (default 11).

    Returns
    -------
    omitted_seqs : list of str
        Each entry is the subsequence that was indirectly omitted (window).
    delta_rgs : list of float
        Each entry is Rg_window - Rg_original for that subsequence/window.
    n_values : list of int
        Each entry is the 1-based start index n of the omitted window
        in the original sequence.
    """
    # --- Encode sequence and run UniRep once ---
    X_jax = one_hot_encode_sequence(seq)         # (L, 26), jnp
    L = X_jax.shape[0]

    # Hidden states: (L, 1900)
    h_states_jax = unirep_apply(unirep_params, X_jax)
    h_states_np = np.asarray(h_states_jax, dtype=np.float32)  # (L, 1900)

    # --- Convert PCA/KRR params to NumPy once ---
    C_np = np.asarray(C, dtype=np.float32)               # (D_pca, 1900)
    mu_np = np.asarray(mu, dtype=np.float32)             # (1900,)
    Z_fit_np = np.asarray(Z_fit, dtype=np.float32)       # (n_train, D_pca)
    dual_np = np.asarray(dual, dtype=np.float32)         # (n_train,)
    gamma_val = float(gamma)
    intercept_val = float(krr_intercept)

    print("Predicting base Rg")
    # --- Original Rg with full mean embedding ---
    h_avg_full = h_states_np.mean(axis=0)                # (1900,)
    rg_original = _krr_predict_from_h_avg_np(
        h_avg_full,
        C_np,
        mu_np,
        Z_fit_np,
        dual_np,
        gamma_val,
        intercept_val,
    )

    omitted_seqs = []
    delta_rgs = []
    n_values = []   # 1-based positions

    # --- Loop over window sizes and positions ---
    # Ensure we never remove ALL positions: w <= L-1
    max_w_eff = min(max_w, max(L - 1, 0))

    for w in range(1, max_w_eff + 1):
        print("Window size:", w)
        # Python indices: start in [0, L-w]
        for start in range(0, L - w + 1):
            # Build a boolean mask for included positions
            mask = np.ones(L, dtype=bool)
            mask[start : start + w] = False

            # Safety: ensure at least one position remains
            if not mask.any():
                continue

            h_avg_omit = h_states_np[mask].mean(axis=0)   # (1900,)

            rg_omit = _krr_predict_from_h_avg_np(
                h_avg_omit,
                C_np,
                mu_np,
                Z_fit_np,
                dual_np,
                gamma_val,
                intercept_val,
            )

            delta_rg = rg_omit - rg_original  # "Rg with omitted layers vs original"

            subseq = seq[start : start + w]
            n_1_based = start + 1

            omitted_seqs.append(subseq)
            delta_rgs.append(float(delta_rg))
            n_values.append(n_1_based)

    return omitted_seqs, delta_rgs, n_values


if __name__ == "__main__":
    # ----------------------------------------------------
    # Big accumulators over ALL proteins
    # ----------------------------------------------------
    all_omitted_seqs = []   # list[str]
    all_delta_rgs = []      # list[float]
    all_n_values = []       # list[int]

    # ----------------------------------------------------
    # Load sequences from training/inliers.csv
    # ----------------------------------------------------
    df = pd.read_csv("training/inliers.csv")

    if "Sequence" not in df.columns:
        raise KeyError("Expected a 'Sequence' column in training/inliers.csv")

    sequences = df["Sequence"].astype(str).tolist()

    # ----------------------------------------------------
    # Loop over every protein and run the window method
    # ----------------------------------------------------
    for i, seq in enumerate(sequences):
        seq = seq.strip()
        if not seq:
            continue  # skip empty sequences just in case

        # Run window-omission analysis on this sequence
        omitted_seqs, delta_rgs, n_values = window_omission_effects(
            seq,
            unirep_params,
            C,
            mu,
            Z_fit,
            dual,
            gamma,
            krr_intercept,
            max_w=11,   # windows of size 1..11
        )

        # Append this protein's results to the global lists
        all_omitted_seqs.extend(omitted_seqs)
        all_delta_rgs.extend(delta_rgs)
        all_n_values.extend(n_values)

        # Optional progress print
        if (i + 1) % 1 == 0:
            print(f"Processed {i + 1} sequences")

    # Save three lists as separate .txt files (one item per line)
    with open("all_omitted_seqs.txt", "w", encoding="utf-8") as f:
        for s in all_omitted_seqs:
            f.write(f"{s}\n")

    with open("all_delta_rgs.txt", "w", encoding="utf-8") as f:
        for v in all_delta_rgs:
            f.write(f"{v}\n")

    with open("all_n_values.txt", "w", encoding="utf-8") as f:
        for n in all_n_values:
            f.write(f"{n}\n")

    
    # Save aggregated results to a single CSV
    assert len(all_omitted_seqs) == len(all_delta_rgs) == len(all_n_values), "Result lists must be same length"

    out_df = pd.DataFrame({
        "omitted_seq": all_omitted_seqs,
        "delta_rg": all_delta_rgs,
        "start_pos": all_n_values,
    })

    out_df.to_csv("window_omissions_all.csv", index=False)
    print(f"Wrote {len(out_df)} rows to window_omissions_all.csv")

    # At this point you have:
    #   all_omitted_seqs : list of all subsequences omitted across the dataset
    #   all_delta_rgs    : corresponding list of delta Rg values
    #   all_n_values     : corresponding list of 1-based start positions