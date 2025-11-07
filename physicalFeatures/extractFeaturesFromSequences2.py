import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import math
import csv

with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    
    sequences = []
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
        counter += 1

# --- Helper functions for patterning metrics (κ, SCD, SHD, π-patterning) ---

def kappa(sequence, charges):
    """
    κ (kappa): Charge segregation index (Das & Pappu 2013)
    sequence: uppercase string, charges: list of +1, -1, or 0
    """
    L = len(sequence)
    if L <= 1:
        return np.nan
    q_sum = np.sum(np.abs(charges))
    if q_sum == 0:
        return 0.0
    mean_q = np.sum(charges) / L
    kappa_val = 0
    for i in range(L):
        for j in range(i+1, L):
            kappa_val += charges[i] * charges[j] * ((j - i) / L)**2
    kappa_val *= (L / (q_sum**2))
    return abs(kappa_val)

def scd(sequence, charges):
    """
    SCD: Sequence charge decoration (Sawle & Ghosh 2015)
    """
    L = len(sequence)
    scd_val = 0
    for i in range(L):
        for j in range(i+1, L):
            scd_val += charges[i] * charges[j] * np.sqrt(j - i)
    return scd_val / L

def shd(sequence, hydros):
    """
    SHD: Sequence hydropathy decoration (Holehouse et al. 2017)
    """
    L = len(sequence)
    shd_val = 0
    for i in range(L):
        for j in range(i+1, L):
            shd_val += hydros[i] * hydros[j] / np.sqrt(j - i)
    return shd_val / L

def pi_patterning(sequence):
    """
    π-patterning: aromatic residue patterning (F, Y, W)
    Similar to SHD but only for aromatics.
    """
    L = len(sequence)
    aromatic = set("FYW")
    h = np.array([1 if aa in aromatic else 0 for aa in sequence])
    if np.sum(h) <= 1:
        return 0.0
    shd_val = 0
    for i in range(L):
        for j in range(i+1, L):
            shd_val += h[i] * h[j] / np.sqrt(j - i)
    return shd_val / L

# --- Core feature extraction ---

def compute_features(seq):
    seq = ''.join([aa for aa in seq.upper() if aa in "ACDEFGHIKLMNPQRSTVWY"])
    if not seq:
        return None
    L = len(seq)
    X = ProteinAnalysis(seq)

    # Residue frequencies
    counts = Counter(seq)
    freq = {aa: counts.get(aa, 0)/L for aa in "ACDEFGHIKLMNPQRSTVWY"}

    # Charge encoding
    charge_map = {'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}
    charges = [charge_map.get(aa, 0) for aa in seq]

    # Hydropathy scale (Kyte & Doolittle)
    kd = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
          'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
          'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
    hydros = [kd.get(aa, 0) for aa in seq]

    # Basic features
    features = [ L,
        X.gravy(),
        X.aromaticity(),
        X.isoelectric_point(),
        (freq["K"] + freq["R"] + freq["H"]),
       (freq["D"] + freq["E"]),
        (freq["K"] + freq["R"] + 0.1*freq["H"]) - (freq["D"] + freq["E"]),
       kappa(seq, charges),
       scd(seq, charges),
        shd(seq, hydros),
        freq["P"], freq["G"],
         -sum(p * math.log(p) for p in freq.values() if p > 0),
        np.mean(charges[:L//2]) - np.mean(charges[L//2:]), pi_patterning(seq)]

    return features

# --- Apply to dataset ---
headers = [
       "Length", "Hydropathy", "Aromaticity", "IsoelectricPt",
        "PosFrac", "NegFrac", "NetCharge", "Kappa", "SCD", "SHD",
        "ProFrac", "GlyFrac", "Entropy", "TerminalChargeAsymmetry",
        "PiPatterning"
    ]
output_file = "protein_features2.csv"
X_phys = [compute_features(seq) for seq in sequences]
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sequence"] + headers)
    for seq, feat in zip(sequences, X_phys):
        x = [seq]
        x.extend(feat)
        writer.writerow(x)

print(f"✅ Wrote {len(sequences)} sequences to {output_file}")