import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

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

inliersUnadjI = unadjI[x]
inlierPhUnadj = inlierPh[inliersUnadjI]

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
                    urea += 1
            others = [reducing, edta, urea, chaps, float(row[34])]
            total = salt_vec + buffer_vec + others
            totals.append(total)
            totalMatrix.append(total)
        elif counter != 1:
            total = [150,2,0,0,0,1.0,0,0,2,1,0,0,7.0]
            totalMatrix.append(total)
        counter += 1

print(len(totalMatrix))
print(len(PhsAdj))
print(len(totals))
print(len(Phs))
print(totalMatrix[0])

outlierIndices =  [123, 136, 151, 158, 171, 185]
x = np.ones(len(PhsAdj), dtype=bool)
x[outlierIndices] = False
totalMatrix = np.array(totalMatrix)
inlierTotals = totalMatrix[x]
inlierTotalUnadj = inlierTotals[inliersUnadjI]

#sanity check

for i in range(len(Phs)):
    if totals[i][12] != Phs[i]:
        print(i)

X_env = np.array(totals, dtype=float)
X_envUnadj = np.array(totalMatrix, dtype = float)

def regressOutBuffer(fileName, X, unadjI, xitem, yitem, title, unadjusted = False):

    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        labels = []
        for row in reader:
            labels.append(row[1])
    labels = np.array(labels[1:], dtype=float)
    if unadjusted:
        labels = labels[unadjI]
    
    #training split
    X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42)

    #Normalize (since we have mixed scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #make the model
    #Trying all variations of alpha, w/5 fold cross validation
    model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    model.fit(X_train_scaled, y_train)

    r2_train = model.score(X_train_scaled, y_train)
    r2_test = model.score(X_test_scaled, y_test)

    print(f"Train R²: {r2_train:.3f}, Test R²: {r2_test:.3f}")

    r2s = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print("CV R² mean:", np.mean(r2s))


    #kernel ridge regression:
    kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.5)
    kr.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = kr.predict(X_train_scaled)
    y_test_pred = kr.predict(X_test_scaled)

    # R² scores
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    #for gamma in [0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        #kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma)
        #kr.fit(X_train_scaled, y_train)
        #print(f"γ={gamma:.2f}, train={kr.score(X_train_scaled, y_train):.3f}, test={kr.score(X_test_scaled, y_test):.3f}")
    
    for alpha in [0.01, 0.1, 1, 10]:
        for gamma in [ 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            kr = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
            kr.fit(X_train_scaled, y_train)
            print(f"alpha={alpha:.2f}, γ={gamma:.4f}, train={kr.score(X_train_scaled, y_train):.3f}, test={kr.score(X_test_scaled, y_test):.3f}")
    print("Kernel Ridge Regression:")
    print(f"Train R²: {r2_train:.3f}, Test R²: {r2_test:.3f}")


def makeResiduals(fileName, X, unadjI, xitem='', yitem='', title='', unadjusted = False):
    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        labels = []
        for row in reader:
            labels.append(row[1])
    labels = np.array(labels[1:], dtype=float)
    if unadjusted:
        labels = labels[unadjI]
    kr = KernelRidge(alpha=10.0, kernel='rbf', gamma=0.0001)
    kr2 = KernelRidge(alpha=10.0, kernel='rbf', gamma=0.01)
    kr2.fit(X, labels)
    kr.fit(X, labels)
    r2 = kr.score(X, labels)
    r22 = kr2.score(X, labels)
    labelsPred2 = kr2.predict(X)
    labelsPred = kr.predict(X)
    residuals = labels - labelsPred
    residuals2 = labels - labelsPred
    #print(f"{xitem} explains {r2:.2%} of variance in {yitem}")
    #print(f"{xitem} explains {r22:.2%} of variance in {yitem}")

    return residuals2


    

#regressOutBuffer('data/allNormalized.csv', Phs, unadjI, 'Buffer Solution', 'Rg (nm)', 'Buffer solution with normalized Rgs', True)


#we want two csv files, one with outliers, one without them, and the Rgs for each type of variation we make

#the one with all the points:
with open('data/allRaw.csv', newline='') as f:
    Rgs0 = []
    sequences = []
    reader = csv.reader(f)

    counter = 0
    for row in reader:
        if counter != 0:
            Rgs0.append(row[1])
            sequences.append(row[0])
        counter += 1
Rgs0 = np.array(Rgs0)
def readCSVfile(fileName):
    with open(fileName, newline='') as f:
        Rgs = []
        reader = csv.reader(f)

        counter = 0
        for row in reader:
            if counter != 0:
                Rgs.append(row[1])
            counter += 1
    return np.array(Rgs)
Rgs1 = readCSVfile('data/allNormalized.csv')
Rgs2 = readCSVfile('data/allNormalizedWithInliers.csv')
Rgs3 = readCSVfile('data/allNormalizedNaive.csv')
Rgs4 = makeResiduals('data/allRaw.csv', PhsAdj, unadjI)
Rgs5 = makeResiduals('data/allNormalized.csv', PhsAdj, unadjI)
Rgs6 = makeResiduals('data/allNormalizedNaive.csv', PhsAdj, unadjI)
Rgs7 = makeResiduals('data/allNormalizedWithInliers.csv', PhsAdj, unadjI)
Rgs8 = makeResiduals('data/allRaw.csv', totalMatrix, unadjI)
Rgs9 = makeResiduals('data/allNormalized.csv', totalMatrix, unadjI)
Rgs10 = makeResiduals('data/allNormalizedNaive.csv', totalMatrix, unadjI)
Rgs11 = makeResiduals('data/allNormalizedWithInliers.csv', totalMatrix, unadjI)


Rgs12 = Rgs0.copy()
unadjI = unadjI.tolist()
Rgs12[unadjI] = makeResiduals('data/allRaw.csv', Phs, unadjI, unadjusted=True)
Rgs13 = Rgs1.copy()
Rgs13[unadjI] = makeResiduals('data/allNormalized.csv', Phs, unadjI, unadjusted=True)
Rgs14 = Rgs2.copy()
Rgs14[unadjI]= makeResiduals('data/allNormalizedNaive.csv', Phs, unadjI, unadjusted=True)
Rgs15 = Rgs3.copy()
Rgs15[unadjI] = makeResiduals('data/allNormalizedWithInliers.csv', Phs, unadjI, unadjusted = True)
Rgs16 = Rgs0.copy()
Rgs16[unadjI] = makeResiduals('data/allRaw.csv', totals, unadjI, unadjusted=True)
Rgs17 = Rgs1.copy()
Rgs17[unadjI] = makeResiduals('data/allNormalized.csv', totals, unadjI, unadjusted=True)
Rgs18 = Rgs2.copy()
Rgs18[unadjI]= makeResiduals('data/allNormalizedNaive.csv', totals, unadjI, unadjusted=True)
Rgs19 = Rgs3.copy()
Rgs19[unadjI] = makeResiduals('data/allNormalizedWithInliers.csv', totals, unadjI, unadjusted = True)



with open('all_points.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    labels = ['Sequence', 'Rg (nm)', 'Rg normalized w/0.427','Rg normalized w/0.5 (nm)', 'Rg normalized w/0.418 (nm)', 
    'Rg w/pH regressed out', 'Rg normalized w/0.427 w/pH regressed out','Rg normalized w/0.5 w/pH regressed out', 'Rg normalized w/0.418 w/pH regressed out',
     'Rg w/buffer regressed out', 'Rg normalized w/0.427 w/buffer regressed out','Rg normalized w/0.5 w/buffer regressed out', 'Rg normalized w/0.418 w/buffer regressed out', 
     'Rg w/experimental pH regressed out', 'Rg normalized w/0.427 w/experimental pH regressed out','Rg normalized w/0.5 w/experimental pH regressed out', 'Rg normalized w/0.418 w/experimental pH regressed out',
      'Rg w/experimental buffer regressed out', 'Rg normalized w/0.427 w/experimental buffer regressed out','Rg normalized w/0.5 w/experimental buffer regressed out', 'Rg normalized w/0.418 w/experimental buffer regressed out']
    writer.writerow(labels)
    for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u in zip(sequences, Rgs0, Rgs1, Rgs2, Rgs3, Rgs4, Rgs5, Rgs6, Rgs7, Rgs8, Rgs9, Rgs10, Rgs11, Rgs12, Rgs13, Rgs14, Rgs15, Rgs16, Rgs17, Rgs18, Rgs19):
        y = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u]
        writer.writerow(y)

with open('data/inliersRaw.csv', newline='') as f:
    rgs0 = []
    Sequences = []
    reader = csv.reader(f)

    counter = 0
    for row in reader:
        if counter != 0:
            rgs0.append(row[1])
            Sequences.append(row[0])
        counter += 1
rgs0 = np.array(rgs0)
rgs1 = readCSVfile('data/inliersNormalizedWithAll.csv')
rgs3 = readCSVfile('data/inliersNormalized.csv')
rgs2 = readCSVfile('data/inliersNormalizedNaive.csv')

rgs4 = makeResiduals('data/inliersRaw.csv', inlierPh, unadjI)
rgs5 = makeResiduals('data/inliersNormalizedWithAll.csv', inlierPh, unadjI)
rgs7 = makeResiduals('data/inliersNormalized.csv', inlierPh, unadjI)
rgs6 = makeResiduals('data/inliersNormalizedNaive.csv', inlierPh, unadjI)

rgs8 = makeResiduals('data/inliersRaw.csv', inlierTotals, unadjI)
rgs9 = makeResiduals('data/inliersNormalizedWithAll.csv', inlierTotals, unadjI)
rgs11 = makeResiduals('data/inliersNormalized.csv', inlierTotals, unadjI)
rgs10 = makeResiduals('data/inliersNormalizedNaive.csv', inlierTotals, unadjI)

rgs12 = rgs0.copy()
rgs12[inliersUnadjI] = makeResiduals('data/inliersRaw.csv', inlierPhUnadj, inliersUnadjI, unadjusted=True)
rgs13 = rgs1.copy()
rgs13[inliersUnadjI] = makeResiduals('data/inliersNormalizedWithAll.csv', inlierPhUnadj, inliersUnadjI, unadjusted=True)
rgs14 = rgs2.copy()
rgs14[inliersUnadjI]= makeResiduals('data/inliersNormalizedNaive.csv', inlierPhUnadj, inliersUnadjI, unadjusted=True)
rgs15 = rgs3.copy()
rgs15[inliersUnadjI] = makeResiduals('data/inliersNormalized.csv', inlierPhUnadj, inliersUnadjI, unadjusted = True)

rgs16 = rgs0.copy()
rgs16[inliersUnadjI] = makeResiduals('data/inliersRaw.csv', inlierTotalUnadj, inliersUnadjI, unadjusted=True)
rgs17 = rgs1.copy()
rgs17[inliersUnadjI] = makeResiduals('data/inliersNormalizedWithAll.csv', inlierTotalUnadj, inliersUnadjI, unadjusted=True)
rgs18 = rgs2.copy()
rgs18[inliersUnadjI]= makeResiduals('data/inliersNormalizedNaive.csv', inlierTotalUnadj, inliersUnadjI, unadjusted=True)
rgs19 = rgs3.copy()
rgs19[inliersUnadjI] = makeResiduals('data/inliersNormalized.csv', inlierTotalUnadj, inliersUnadjI, unadjusted = True)


with open('inliers.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    labels = ['Sequence', 'Rg (nm)', 'Rg normalized w/0.427','Rg normalized w/0.5 (nm)', 'Rg normalized w/0.418 (nm)', 
    'Rg w/pH regressed out', 'Rg normalized w/0.427 w/pH regressed out','Rg normalized w/0.5 w/pH regressed out', 'Rg normalized w/0.418 w/pH regressed out',
     'Rg w/buffer regressed out', 'Rg normalized w/0.427 w/buffer regressed out','Rg normalized w/0.5 w/buffer regressed out', 'Rg normalized w/0.418 w/buffer regressed out', 
     'Rg w/experimental pH regressed out', 'Rg normalized w/0.427 w/experimental pH regressed out','Rg normalized w/0.5 w/experimental pH regressed out', 'Rg normalized w/0.418 w/experimental pH regressed out',
      'Rg w/experimental buffer regressed out', 'Rg normalized w/0.427 w/experimental buffer regressed out','Rg normalized w/0.5 w/experimental buffer regressed out', 'Rg normalized w/0.418 w/experimental buffer regressed out']
    writer.writerow(labels)
    for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u in zip(Sequences, rgs0, rgs1, rgs2, rgs3, rgs4, rgs5, rgs6, rgs7, rgs8, rgs9, rgs10, rgs11, rgs12, rgs13, rgs14, rgs15, rgs16, rgs17, rgs18, rgs19):
        y = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u]
        writer.writerow(y)