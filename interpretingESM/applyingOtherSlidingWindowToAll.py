import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
import matplotlib.pyplot as plt
import csv 
from tryingAnotherTypeOfSlidingWindow import sliding_occlusion_rg
import pickle

tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model.eval()
regr_model = joblib.load('esm_gpr.joblib')
pca = joblib.load('esm_pca.joblib')



sequences = []
with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            sequences.append(row[0])
        counter += 1

allEffects = []
allFragments = []
allIndices = []
for i in range(1, 11):
    effectsList = []
    baselines = []
    fragmentsList = []
    indicesList = []

    for s in sequences:
        effects, baseleine, fragments, indices = sliding_occlusion_rg(s, regr_model, pca, window = i)
        effectsList.append(effects)
        baselines.append(baseleine)
        fragmentsList.append(fragments)
        indicesList.append(indices)
    allEffects.append(effects)
    allFragments.append(fragmentsList)
    allIndices.append(indicesList)

with open("allEffects2.pkl", "wb") as f:
    pickle.dump(allEffects, f)
with open("allMaskedPositions2.pkl", "wb") as f:
    pickle.dump(allIndices, f)
with open("allFragments2.pkl", "wb") as f:
    pickle.dump(allFragments, f)
with open("baselines.pkl", "wb") as f:
    pickle.dump(baselines, f)