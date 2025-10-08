import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from writeCsv import write_csv
import matplotlib.pyplot as plt

Rgs = []
Phs = []
RgsAdj = []
PhsAdj = []
unadjI = []

def regressOutFrom(fileName, features, AdjI, xitem, yitem):
    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        labels = []
        for row in reader:
            labels.append(row[1])
    labels = np.array(labels[1:], dtype=float)
    model = LinearRegression().fit(features, labels)
    r2 = model.score(features, labels)
    labelsPred = model.predict(features)
    residuals = labels - labelsPred

    print(f"{xitem} explains {r2:.2%} of variance in {yitem}")
    plt.figure(figsize=(6,4))
    plt.scatter(features, labels, label=f"Observed {yitem}", color="gray")
    plt.plot(features, labelsPred, label="Linear fit", color="red")
    plt.xlabel(f"{xitem}")
    plt.ylabel(f"{yitem} (nm)")
    plt.title(f"{yitem} vs {xitem} with linear fit")
    plt.legend()
    plt.show()

    # Residuals plot
    plt.figure(figsize=(6,4))
    plt.scatter(features, residuals, color="blue")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel(f"{xitem}")
    plt.ylabel(f"Residual ({xitem}-corrected {yitem})")
    plt.title(f"Residuals after {xitem} regression (should look random)")
    plt.show()
    return residuals, r2

with open('data/rawData.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[3] != '' and row[34]!= '' and row[2] != '':
            Phs.append(row[34])
            Rgs.append(row[3])
            RgsAdj.append(row[3])
            PhsAdj.append(row[34])
            unadjI.append(1)
        elif row[3] != '' and row[2] != '':
            RgsAdj.append(row[3])
            #hard code in 7.0
            PhsAdj.append(7.0)
            unadjI.append(0)
unadjI = np.array(unadjI)
AdjI = ~unadjI
Phs = np.array(Phs[1:], dtype= float).reshape(-1, 1)
PhsAdj = np.array(PhsAdj[1:], dtype = float).reshape(-1, 1)
Rgs = np.array(Rgs[1:], dtype=float)
RgsAdj = np.array(Rgs[1:], dtype=float)

#Fit to linear regression
pHResiduals1, phR21 = regressOutFrom('data/allRaw.csv', PhsAdj, AdjI, 'pH', 'Unnorm Rg')

#Try with normalized w/0.427
pHResiduals2, pHR22 = regressOutFrom('data/allNormalized.csv', PhsAdj, AdjI, 'pH', 'Norm Rg w/0.427')

#Try Normalized w/0.418
pHResiduals3, pHR23 = regressOutFrom('data/allNormalizedWithInliers.csv', PhsAdj, AdjI, 'pH', 'Norm Rg w/0.418')

#Try Normalized w/0.5
pHResiduals4, pHR24 = regressOutFrom('data/allNormalizedNaive.csv', PhsAdj, AdjI, 'pH', 'Norm Rg w/0.5')

#Adjust for Outliers
outlierIndices =  [123, 136, 151, 158, 171, 185]
x = np.ones(len(PhsAdj), dtype=bool)
x[outlierIndices] = False
inlierPh = PhsAdj[x]
print(len(inlierPh))

pHResiduals5, pHR25 = regressOutFrom('data/inliersRaw.csv', inlierPh, AdjI, 'pH', 'Unnorm Rg')
pHResiduals6, pHR26 = regressOutFrom('data/inliersNormalized.csv', inlierPh, AdjI, 'pH', 'Norm Rg w/0/418')
pHResiduals7, pHR27 = regressOutFrom('data/inliersNormalizedWithAll.csv', inlierPh, AdjI, 'pH', 'Norm Rg w/0.427')
pHResiduals8, pHR28 = regressOutFrom('data/inliersNormalizedNaive.csv', inlierPh, AdjI, 'pH', 'Norm Rg w/0.5')
