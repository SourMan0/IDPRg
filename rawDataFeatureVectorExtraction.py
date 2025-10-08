import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# === Amino acid properties ===
# Kyte-Doolittle hydropathy scale
hydropathy_index = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Standard amino acids
aa_list = list('ACDEFGHIKLMNPQRSTVWY')

# pKa values for ionizable groups
pKa_values = {
    'D': 3.9,   # Aspartic acid (carboxyl)
    'E': 4.3,   # Glutamic acid (carboxyl)
    'C': 8.3,   # Cysteine (thiol)
    'Y': 10.1,  # Tyrosine (phenolic)
    'H': 6.0,   # Histidine (imidazole)
    'K': 10.5,  # Lysine (amino)
    'R': 12.5,  # Arginine (guanidinium)
    'N_term': 8.0,  # N-terminus
    'C_term': 3.1   # C-terminus
}

# Ionic charges for common buffer components (for ionic strength calculation)
# Format: {ion: (concentration_multiplier, charge)}
ion_charges = {
    '[NaCl] mM': [(1, 1), (1, -1)],  # Na+ and Cl-
    '[Na3PO4] mM': [(3, 1), (1, -3)],  # 3Na+ and PO4^3-
    '[Na2HPO4] mM': [(2, 1), (1, -2)],  # 2Na+ and HPO4^2-
    '[KH2PO4] mM': [(1, 1), (1, -2)],  # K+ and H2PO4^-
    '[KCl] mM': [(1, 1), (1, -1)],  # K+ and Cl-
    '[MgCl2] mM': [(1, 2), (2, -1)],  # Mg^2+ and 2Cl-
    '[CaCl2] mM': [(1, 2), (2, -1)],  # Ca^2+ and 2Cl-
    '[NaNO3] mM': [(1, 1), (1, -1)],  # Na+ and NO3-
    '[Na2SO4] mM': [(2, 1), (1, -2)],  # 2Na+ and SO4^2-
    '[(NH4)2SO4] mM': [(2, 1), (1, -2)],  # 2NH4+ and SO4^2-
    '[NaF] mM': [(1, 1), (1, -1)],  # Na+ and F-
    '[Tris-HCl] mM': [(1, 1), (1, -1)],  # Tris-H+ and Cl-
    '[HEPES] mM': [(0.5, 1), (0.5, -1)],  # Partial ionization
    '[MES] mM': [(0.5, 1), (0.5, -1)],  # Partial ionization
    '[MOPS] mM': [(0.5, 1), (0.5, -1)],  # Partial ionization
}

def calculate_charge_at_pH(sequence: str, pH: float) -> float:
    """
    Calculate net charge of protein at given pH using Henderson-Hasselbalch equation.
    """
    if not sequence or pH <= 0:
        return 0.0
    
    sequence = sequence.upper().strip()
    charge = 0.0
    
    # Count ionizable residues
    for aa in sequence:
        if aa in pKa_values:
            pKa = pKa_values[aa]
            
            # Acidic groups (lose proton as pH increases)
            if aa in ['D', 'E', 'C', 'Y']:
                charge -= 1 / (1 + 10**(pKa - pH))
            
            # Basic groups (gain proton as pH decreases)
            elif aa in ['H', 'K', 'R']:
                charge += 1 / (1 + 10**(pH - pKa))
    
    # Terminal charges
    # N-terminus (NH3+ -> NH2)
    charge += 1 / (1 + 10**(pH - pKa_values['N_term']))
    
    # C-terminus (COOH -> COO-)
    charge -= 1 / (1 + 10**(pKa_values['C_term'] - pH))
    
    return charge

def calculate_ionic_strength(buffer_dict: Dict[str, float]) -> float:
    """
    Calculate ionic strength from buffer composition.
    I = 0.5 * Σ(ci * zi^2) where ci is molar concentration and zi is charge
    """
    ionic_strength = 0.0
    
    for buffer_col, concentration in buffer_dict.items():
        if concentration > 0 and buffer_col in ion_charges:
            # Get ion contributions for this buffer component
            for multiplier, charge in ion_charges[buffer_col]:
                # Convert mM to M and calculate contribution
                conc_M = (concentration * multiplier) / 1000.0
                ionic_strength += conc_M * (charge ** 2)
    
    # For other buffers not in the dictionary, assume monovalent salt
    for buffer_col, concentration in buffer_dict.items():
        if concentration > 0 and buffer_col not in ion_charges:
            # Assume 1:1 monovalent salt as approximation
            conc_M = concentration / 1000.0
            ionic_strength += 2 * conc_M * (1 ** 2)  # Two ions each with charge ±1
    
    return ionic_strength * 0.5

def compute_hydropathy_stats(sequence: str) -> Tuple[float, float, float, float]:
    """
    Compute hydropathy statistics: mean, variance, min, max
    """
    if not sequence:
        return 0.0, 0.0, 0.0, 0.0
    
    sequence = sequence.upper().strip()
    hydro_values = [hydropathy_index.get(aa, 0.0) for aa in sequence if aa in hydropathy_index]
    
    if not hydro_values:
        return 0.0, 0.0, 0.0, 0.0
    
    return (
        np.mean(hydro_values),
        np.var(hydro_values),
        np.min(hydro_values),
        np.max(hydro_values)
    )

def compute_aa_composition(sequence: str) -> Dict[str, float]:
    """
    Compute amino acid composition (frequency)
    """
    sequence = sequence.upper().strip()
    length = len(sequence)
    
    if length == 0:
        return {aa: 0.0 for aa in aa_list}
    
    aa_counts = {aa: sequence.count(aa) for aa in aa_list}
    return {aa: count / length for aa, count in aa_counts.items()}

def compute_dipeptide_composition(sequence: str, top_n: int = 10) -> Dict[str, float]:
    """
    Compute dipeptide composition for most common dipeptides
    """
    sequence = sequence.upper().strip()
    length = len(sequence) - 1
    
    if length <= 0:
        return {}
    
    dipeptide_counts = {}
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if all(aa in aa_list for aa in dipeptide):
            dipeptide_counts[dipeptide] = dipeptide_counts.get(dipeptide, 0) + 1
    
    # Get top N most common dipeptides
    if dipeptide_counts:
        sorted_dipeptides = sorted(dipeptide_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {dp: count / length for dp, count in sorted_dipeptides}
    
    return {}

def compute_features(sequence: str, pH: float, buffer_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Compute focused feature vector for a protein sequence
    Only includes: length, charge metrics, hydropathy stats, AA composition groups, disorder, entropy
    """
    sequence = sequence.upper().strip()
    length = len(sequence)
    
    features = {}
    
    # Length
    features['length'] = length
    
    if length == 0:
        # Return zero features for empty sequence
        return features
    
    # === CHARGE FEATURES ===
    net_charge = calculate_charge_at_pH(sequence, pH)
    features['net_charge_per_residue'] = net_charge / length
    
    # Fraction of charged residues
    n_positive = sum(sequence.count(aa) for aa in ['K', 'R', 'H'])
    n_negative = sum(sequence.count(aa) for aa in ['D', 'E'])
    features['frac_charged'] = (n_positive + n_negative) / length
    
    # === HYDROPATHY FEATURES ===
    hydro_mean, hydro_var, hydro_min, hydro_max = compute_hydropathy_stats(sequence)
    features['hydropathy_mean'] = hydro_mean
    features['hydropathy_var'] = hydro_var
    features['hydropathy_range'] = hydro_max - hydro_min
    
    # === AMINO ACID COMPOSITION GROUPS ===
    # Hydrophobic
    hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
    features['frac_hydrophobic'] = sum(sequence.count(aa) for aa in hydrophobic) / length
    
    # Polar
    polar = ['S', 'T', 'N', 'Q', 'Y', 'C']
    features['frac_polar'] = sum(sequence.count(aa) for aa in polar) / length
    
    # Aromatic
    aromatic = ['F', 'Y', 'W']
    features['frac_aromatic'] = sum(sequence.count(aa) for aa in aromatic) / length
    
    # Small
    small = ['G', 'A', 'S', 'T', 'C', 'V']
    features['frac_small'] = sum(sequence.count(aa) for aa in small) / length
    
    # === DISORDER FEATURES ===
    disorder_promoting = ['A', 'R', 'G', 'Q', 'S', 'P', 'E', 'K']
    features['frac_disorder_promoting'] = sum(sequence.count(aa) for aa in disorder_promoting) / length
    
    # === SEQUENCE ENTROPY ===
    aa_comp = compute_aa_composition(sequence)
    aa_freqs = [freq for freq in aa_comp.values() if freq > 0]
    if aa_freqs:
        entropy = -sum(f * np.log2(f) for f in aa_freqs if f > 0)
        features['sequence_entropy'] = entropy
    else:
        features['sequence_entropy'] = 0.0
    
    return features

def process_dataset(input_file: str, output_file: str = 'protein_features.csv'):
    """
    Process the entire dataset and extract features
    """
    print(f"Loading dataset from {input_file}...")
    
    # Try different encoding if UTF-8 fails
    df = pd.read_csv("trainingData.csv", header=0)
    print(df.columns.tolist())
    print(df.head(3))
    
    print(f"Loaded {len(df)} entries")
    
    # Identify buffer columns
    buffer_cols = [col for col in df.columns if '[' in col and 'mM' in col]
    print(f"Found {len(buffer_cols)} buffer columns")
    
    # Define feature names (must match compute_features order)
    aa_order = list(hydropathy_index.keys())  # same order as compute_features
    base_features = [
        "length", "net_charge", "frac_pos", "frac_neg",
        "hydropathy_mean", "hydropathy_var", "ionic_strength"
    ]
    aa_features = [f"AA_{aa}" for aa in aa_order]
    feature_names = base_features + aa_features
    
    # Process each entry
    all_features = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        # --- Sequence ---
        seq = str(row.get('Experimental Sequence', '')).strip()
        if not seq or seq.lower() == 'nan':
            continue
        
        # --- pH (robust detection) ---
        pH_value = None
        for ph_col in ['pH', 'Ph', 'PH', 'ph']:
            if ph_col in df.columns:
                try:
                    pH_value = float(row[ph_col])
                    break
                except:
                    continue
        if pH_value is None or pd.isna(pH_value):
            pH_value = 7.0  # default
        
        # --- Buffers ---
        buffer_dict = {}
        for col in buffer_cols:
            val = row[col]
            if pd.notna(val):
                try:
                    buffer_dict[col] = float(val)
                except:
                    buffer_dict[col] = 0.0
            else:
                buffer_dict[col] = 0.0
        
        # --- Compute features ---
        vec = compute_features(seq, pH_value, buffer_dict)

        # Start with the feature dict directly
        feature_dict = vec.copy()

        # --- Target + metadata ---
        feature_dict['Rg_nm'] = row.get('Rg (nm)', np.nan)
        feature_dict['protein_name'] = row.get('Protein Name', '')
        
        all_features.append(feature_dict)
        valid_indices.append(idx)
    
    # --- Assemble DataFrame ---
    feature_df = pd.DataFrame(all_features, index=valid_indices)
    
    # --- Report ---
    print(f"\nProcessed {len(feature_df)} valid entries")
    if 'Rg_nm' in feature_df.columns:
        print(f"Entries with Rg values: {feature_df['Rg_nm'].notna().sum()}")
    print(f"Number of features: {len(feature_names)}")
    
    # --- Missing values ---
    missing_counts = feature_df.isnull().sum()
    if missing_counts.any():
        print("\nColumns with missing values:")
        print(missing_counts[missing_counts > 0])
    
    # --- Save ---
    feature_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to {output_file}")
    
    return feature_df

# === Main execution ===
if __name__ == "__main__":
    # Process the dataset
    input_file = "IDPRg/rawData.csv"
    output_file = "rawDataProteinFeatureVectors.csv"
    
    try:
        feature_df = process_dataset(input_file, output_file)
        
        print("\n" + "="*50)
        print("Feature extraction completed successfully!")
        print("="*50)
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()