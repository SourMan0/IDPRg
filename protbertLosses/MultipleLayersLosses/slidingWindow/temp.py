import pickle
import numpy as np

eff = pickle.load(open("protbert_sliding_results_top5/model_1/effects_by_window.pkl","rb"))

# Take window=4 for example
w = eff[3]

print("Min effect:", np.min([e.min() for e in w]))
print("Max effect:", np.max([e.max() for e in w]))
print("Mean effect:", np.mean([e.mean() for e in w]))
