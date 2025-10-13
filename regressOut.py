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

def regressOutFrom(fileName, features, unAdjI, xitem, yitem, unadjusted = False):
    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        labels = []
        for row in reader:
            labels.append(row[1])
    labels = np.array(labels[1:], dtype=float)
    if unadjusted:
        labels = labels[unAdjI]
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
            unadjI.append(True)
        elif row[3] != '' and row[2] != '':
            RgsAdj.append(row[3])
            #hard code in 7.0
            PhsAdj.append(7.0)
            unadjI.append(False)
unadjI = np.array(unadjI[1:])
AdjI = ~unadjI
Phs = np.array(Phs[1:], dtype= float).reshape(-1, 1)
PhsAdj = np.array(PhsAdj[1:], dtype = float).reshape(-1, 1)
Rgs = np.array(Rgs[1:], dtype=float)
RgsAdj = np.array(Rgs[1:], dtype=float)

#Fit to linear regression
#pHResiduals1, phR21 = regressOutFrom('data/allRaw.csv', PhsAdj, AdjI, 'pH', 'Unnorm Rg')

#Try with normalized w/0.427
#pHResiduals2, pHR22 = regressOutFrom('data/allNormalized.csv', PhsAdj, AdjI, 'pH', 'Norm Rg w/0.427')

#Try Normalized w/0.418
#pHResiduals3, pHR23 = regressOutFrom('data/allNormalizedWithInliers.csv', PhsAdj, AdjI, 'pH', 'Norm Rg w/0.418')

#Try Normalized w/0.5
#pHResiduals4, pHR24 = regressOutFrom('data/allNormalizedNaive.csv', PhsAdj, AdjI, 'pH', 'Norm Rg w/0.5')

#Adjust for Outliers
outlierIndices =  [123, 136, 151, 158, 171, 185]
x = np.ones(len(PhsAdj), dtype=bool)
x[outlierIndices] = False
inlierPh = PhsAdj[x]
print(len(inlierPh))

#pHResiduals5, pHR25 = regressOutFrom('data/inliersRaw.csv', inlierPh, AdjI, 'pH', 'Unnorm Rg')
#pHResiduals6, pHR26 = regressOutFrom('data/inliersNormalized.csv', inlierPh, AdjI, 'pH', 'Norm Rg w/0/418')
#pHResiduals7, pHR27 = regressOutFrom('data/inliersNormalizedWithAll.csv', inlierPh, AdjI, 'pH', 'Norm Rg w/0.427')
#pHResiduals8, pHR28 = regressOutFrom('data/inliersNormalizedNaive.csv', inlierPh, AdjI, 'pH', 'Norm Rg w/0.5')


#Now let's take a look at just the unadjusted points (ie. the experimental ones)

#phResiduals9, PhR29 = regressOutFrom('data/allRaw.csv', Phs, unadjI, 'pH', 'Unnorm Rg', unadjusted=True)
#phResiduals10, PhR210 = regressOutFrom('data/allNormalized.csv', Phs, unadjI, 'pH', 'Norm Rg w/0.427', True)
#phResiduals11, PhR211 = regressOutFrom('data/allNormalizedWithInliers.csv', Phs, unadjI, 'pH', 'Norm Rg w/0.418', True)
#phResiduals12, PhR212 = regressOutFrom('data/allNormalizedNaive.csv', Phs, unadjI, 'pH', 'Norm Rg w/0.5', True)

#Lets looks at the inliers now

#inliersUnadjI = unadjI[x]
#inlierPhUnadj = inlierPh[inliersUnadjI]

#pHResiduals13, pHR213 = regressOutFrom('data/inliersRaw.csv', inlierPhUnadj, inliersUnadjI, 'pH', 'Unnorm Rg', True)
#pHResiduals14, pHR214 = regressOutFrom('data/inliersNormalized.csv', inlierPhUnadj, inliersUnadjI, 'pH', 'Norm Rg w/0/418', True)
#pHResiduals15, pHR215 = regressOutFrom('data/inliersNormalizedWithAll.csv', inlierPhUnadj, inliersUnadjI, 'pH', 'Norm Rg w/0.427', True)
#pHResiduals8, pHR28 = regressOutFrom('data/inliersNormalizedNaive.csv', inlierPhUnadj, inliersUnadjI, 'pH', 'Norm Rg w/0.5', True)

bufferIndices = {"Monovalent Salt": [6, 10, 21, 26], "Divalent Chloride": [16, 20], "Sulfates": [24, 28], "Phosphates": [7,8,9,23,27], "Tris":[14, 15], "Goods":[18, 19, 22, 29, 31], "Reducing Agents":[13, 17, 25], "EDTA": [11], "Urea": [12], "CHAPS": [33], "NaN3": [30], "PMSF": [12]}

totalMatrix = []
totals = []
counter = 1
with open('data/rawData.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        #salts
        if row[6] != '' and counter != 1:
            monovalent = 0
            divalent = 0
            sulfates = 0
            phosphates = 0
            nitrates = 0
            for i in bufferIndices['Monovalent Salt']:
                if i == 21:
                    if row[i] != '0' and row[i] != '':
                        monovalent += float(row[i])
                        nitrates += float(row[i])
                else:
                   if row[i] != '0' and row[i] != '':
                        monovalent += float(row[i])
            for i in bufferIndices['Divalent Chloride']:
                if row[i] != '0' and row[i] != '':
                    divalent += float(row[i])
            for i in bufferIndices["Sulfates"]:
                if row[i] != '0' and row[i] != '':
                    phosphates += float(row[i])
                    monovalent += 2 * float(row[i])
            for i in bufferIndices["Phosphates"]:
                #estimate 1.5 valency for things labelled with "Na3PO4" due to the ambiguity
                if i == 7:
                    if row[i] != '0' and row[i] != '':
                        monovalent += 1.5 * float(row[i])
                        phosphates += float(row[i])
                elif i == 8 or i == 9:
                    if row[i] != '0' and row[i] != '':
                        monovalent += float(row[i])
                        phosphates += float(row[i])
                elif i == 23:
                    #use 1.5 here again
                    if row[i] != '0' and row[i] != '':
                        monovalent += 1.5 * float(row[i])
                        phosphates += float(row[i])
                else:
                    if row[i] != '0' and row[i] != '':
                        monovalent += 2 * float(row[i])
                        phosphates += float(row[i])
            salt_vec = [monovalent, divalent, sulfates, phosphates, nitrates]  

            tris = 0
            goods = 0
            phos = 0
            for i in bufferIndices['Tris']:
                if row[i] != '0' and row[i] != '':
                    tris += float(row[i])
            for i in bufferIndices['Goods']:
                if row[i] != '0' and row[i] != '':
                    goods += float(row[i])
            #if it's a buffer, better to keep it as such
            if goods == 0 and tris == 0 and salt_vec[3] != 0:
                phos = phosphates
                salt_vec[3] = 0
            s = goods + tris + phos
            if s != 0:
                buffer_vec = [goods / s, tris / s, phos / s]
            else:
                buffer_vec = [0, 0, 0]
                

            reducing = 0
            edta = 0
            urea = 0
            chaps = 0

            for i in bufferIndices['Reducing Agents']:
                if row[i] != '' and row[i] != '0':
                    reducing += float(row[i])
            for i in bufferIndices['EDTA']:
                if row[i] != '' and row[i] != '0':
                    edta += 1
            for i in bufferIndices['CHAPS']:
                if row[i] != '' and row[i] != '0':
                    chaps += 1
            for i in bufferIndices['Urea']:
                if row[i] != '' and row[i] != '0':
                    urea += float(row[i])
            others = [reducing, edta, urea, chaps, float(row[34])]

            total = salt_vec + buffer_vec + others
            totals.append(total)
            totalMatrix.append(total)
        else:
            total = [150,2,0,0,0,1.0,0,0,2,1,0,0,7.0]
            totalMatrix.append(total)
        counter += 1

print(len(totalMatrix))
print(len(totals))
print(totalMatrix[0])


