import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
import matplotlib.pyplot as plt
import csv 
from tryingAnotherTypeOfSlidingWindow import sliding_occlusion_rg
import pickle
###############################################
# 1. LOAD ESM MODEL (layer 4 is index = 4)
###############################################

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

effectsList = []
baselines = []
fragmentsList = []
indicesList = []

for s in sequences:
    effects, baseleine, fragments, indices = sliding_occlusion_rg(s, regr_model, pca)
    effectsList.append(effects)
    baselines.append(baseleine)
    fragmentsList.append(fragments)
    indicesList.append(indices)

with open("allEffects2.pkl", "wb") as f:
    pickle.dump(effectsList, f)
with open("allMaskedPositions2.pkl", "wb") as f:
    pickle.dump(indicesList, f)
with open("allFragments2.pkl", "wb") as f:
    pickle.dump(fragmentsList, f)
with open("baselines.pkl", "wb") as f:
    pickle.dump(baselines, f)