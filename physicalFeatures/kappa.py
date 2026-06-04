import numpy as np

def charges_from_sequence(seq, pos_set={'K', 'R', 'H'}, neg_set={'D', 'E'}):
    q = []
    for aa in seq:
        if aa in pos_set:
            q.append(1)
        elif aa in neg_set:
            q.append(-1)
        else:
            q.append(0)
    return np.array(q, dtype=int)

def delta_same_sign(charges):
    M = len(charges)
    if M < 2:
        return 0.0
    delta = 0.0

    for i in range(M-1):
        for j in range(i+1, M):
            if charges[i] * charges[j] > 0:
                delta += 1.0 / (j - i)
    return delta 


#Maximum possible arrangement for kappa
def delta_max_for_composition(n_pos, n_neg):

    m = n_pos + n_neg
    if m < 2:
        return 0.0
    
    charges_max = np.array([1] * n_pos + [-1] * n_neg, dtype = int)

    return delta_same_sign(charges_max)

def kappa_simple_from_charges(q_full):
    q_full = np.asarray(q_full, dtype = int)

    charged_mask = q_full != 0

    charges = q_full[charged_mask]

    m = len(charges)

    if m < 2:
        return 0.0
    
    n_pos = np.sum(charges > 0)
    n_neg = np.sum(charges < 0)

    if n_pos == 9 or n_neg == 0:
        return 1.0
    
    delta_seq = delta_same_sign(charges)
    delta_max = delta_max_for_composition(n_pos, n_neg)

    if delta_max < 1e-12:
        return 0.0
    return float(delta_seq / delta_max)

def kappa_simple_from_sequence(seq, pos_set = {'K', 'R', 'H'}, neg_set = {'D', 'E'}):
    q_full = charges_from_sequence(seq, pos_set=pos_set, neg_set=neg_set)
    return kappa_simple_from_charges(q_full)