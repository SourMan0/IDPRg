import csv
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    
    sequences = []
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
        counter += 1
print(len(sequences))
# --- Amino acid property tables ---

#From this paper: https://www.sciencedirect.com/science/article/pii/0022283682905150?via%3Dihub
#Looking at the hydropathic character of a protein

#Degree of hydrophobicity

hydropathy = {  # Kyte–Doolittle
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
    'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2
}
mass = {  # average residue mass in Da
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2, 'Q': 146.2,
    'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2, 'L': 131.2, 'K': 146.2,
    'M': 149.2, 'F': 165.2, 'P': 115.1, 'S': 105.1, 'T': 119.1, 'W': 204.2,
    'Y': 181.2, 'V': 117.1
}
charge_dict = {'K': 1, 'R': 1, 'H': 1, 'D': -1, 'E': -1}  # others = 0

# Define residue groups
hydrophobic = set("AVLIMFWY")
polar = set("STNQ")
charged_pos = set("KRH")
charged_neg = set("DE")
aromatic = set("FWY")
special = {'P', 'G'}
#For calculating Patterning features
def patterning_features(seq):
    seq = [aa for aa in seq if aa in hydropathy]
    L = len(seq)
    if L < 2:
        return 0, 0, 0

    q = np.array([charge_dict.get(aa, 0) for aa in seq])
    hydro = np.array([hydropathy[aa] for aa in seq])

    # SCD
    i, j = np.triu_indices(L, k=1)
    dist = np.sqrt(j - i)
    scd = np.sum(q[i] * q[j] / dist) / L

    # SHD
    shd = np.sum((hydro[i] + hydro[j]) / (2 * dist)) / L

    # κ (charge segregation)
    # Normalize to 0–1 range, approximated per Das & Pappu (2013)
    charge_pairs = (q[i] * q[j])
    same_sign = charge_pairs > 0
    opp_sign = charge_pairs < 0
    if np.sum(opp_sign) + np.sum(same_sign) == 0:
        kappa = 0
    else:
        kappa = np.sum(np.abs(j[same_sign] - i[same_sign])) / (
            np.sum(np.abs(j - i)) if np.sum(np.abs(j - i)) else 1
        )

    return kappa, scd, shd
# --- Helper: compute features for one sequence ---
def seq_features(seq):
    seq2 = seq.upper()
    seq = [aa for aa in seq2 if aa in hydropathy]  # filter unknowns
    L = len(seq)
    if L == 0:
        return np.zeros(15)

    # 1. Composition
    f_hydro = sum(aa in hydrophobic for aa in seq) / L
    f_polar = sum(aa in polar for aa in seq) / L
    f_arom = sum(aa in aromatic for aa in seq) / L
    f_pro = seq.count('P') / L
    f_gly = seq.count('G') / L
    f_charged = sum(aa in charged_pos | charged_neg for aa in seq) / L

    # 2. Charge features
    n_pos = sum(aa in charged_pos for aa in seq)
    n_neg = sum(aa in charged_neg for aa in seq)
    ncpr = (n_pos - n_neg) / L
    fcr = (n_pos + n_neg) / L
    charge_asym = abs((n_pos - n_neg)) / (n_pos + n_neg) if (n_pos + n_neg) else 0

    # 3. Hydropathy
    hydro_values = np.array([hydropathy[aa] for aa in seq])
    mean_hydro = np.mean(hydro_values)
    var_hydro = np.var(hydro_values)

    # 4. Global
    # Weight in grams per mole
    mol_wt = ProteinAnalysis(seq2).molecular_weight()
    logL = np.log(L)

    # 5. Composite ratios
    hydro_fcr = mean_hydro / (fcr + 1e-6)
    arom_hydro = f_arom / (f_hydro + 1e-6)

    
    x = [
        f_hydro, f_polar, f_arom, f_pro, f_gly, f_charged,
        ncpr, fcr, charge_asym,
        mean_hydro, var_hydro,
        mol_wt, logL,
        hydro_fcr, arom_hydro
    ]
    #For patterning features
    x.extend(list(patterning_features(seq)))
    return np.array(x)

# --- Run over all sequences ---
X_phys = [seq_features(seq) for seq in sequences]


headers = headers = [
    "Length", "MolWt", "Hydropathy", "Aromaticity", "Instability",
    "Flexibility", "IsoelectricPt", "PosFrac", "NegFrac",
    "NetCharge", "Kappa", "SCD", "SHD"
]
output_file = "protein_features.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sequence"] + headers)
    for seq, feat in zip(sequences, X_phys):
        x = [seq]
        x.extend(feat)
        writer.writerow(x)

print(f"✅ Wrote {len(sequences)} sequences to {output_file}")