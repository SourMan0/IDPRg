import csv
import numpy as np
from testingTheSlidingWindow import sliding_mask_effect
import joblib
from transformers import AutoTokenizer, AutoModel
import pickle


model = joblib.load('esm_gpr.joblib')
pca = joblib.load('esm_pca.joblib')

model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name)


allEffects = []
allMaskedPositions = []
allFragments = []

sequences = []

with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
        counter += 1


lens = []
for s in sequences:
    lens.append(len(s))
print(min(lens))



windowSizes = list(range(1, 11))

for i in windowSizes:
    allEffects.append([])
    allMaskedPositions.append([])
    allFragments.append([])


for s in sequences:
    for w in windowSizes:
        effects, masked_positions, masked_fragments = sliding_mask_effect(s, esm_model, tokenizer, pca, model, window = w)
        allEffects[w - 1].append(effects)
        allFragments[w - 1].append(masked_fragments)
        allMaskedPositions[w - 1].append(masked_positions)


with open("allEffects.pkl", "wb") as f:
    pickle.dump(allEffects, f)
with open("allMaskedPositions.pkl", "wb") as f:
    pickle.dump(allMaskedPositions, f)
with open("allFragments.pkl", "wb") as f:
    pickle.dump(allFragments, f)



