#Move to parent directory to remake csvs 
import numpy as np
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from doAllRegressions import evaluate_models_rmse

features = {}
layers6 = range(7)
PCAvals = [190, 100, 50, 20, 10]

for l in layers6:
    features[l] = {}
    for p in PCAvals:
        features[l][p] = np.load(f'6esmPCA/layer{l}_pca{p}.npy', allow_pickle=True)
layers6 = range(7)
features12 = {}
layers12 = range(13)
for l in layers12:
    features12[l] = {}
    for p in PCAvals:
        features12[l][p] = np.load(f'12esmPCA/layer{l}_pca{p}.npy', allow_pickle=True)
layers12 = range(13)
labels = []
inlierLabels = []
labelHeaders = ['Sequence', 'Rg (nm)', 'Rg normalized w/0.427','Rg normalized w/0.5 (nm)', 'Rg normalized w/0.418 (nm)', 
    'Rg w/pH regressed out', 'Rg normalized w/0.427 w/pH regressed out','Rg normalized w/0.5 w/pH regressed out', 'Rg normalized w/0.418 w/pH regressed out',
     'Rg w/buffer regressed out', 'Rg normalized w/0.427 w/buffer regressed out','Rg normalized w/0.5 w/buffer regressed out', 'Rg normalized w/0.418 w/buffer regressed out', 
     'Rg w/experimental pH regressed out', 'Rg normalized w/0.427 w/experimental pH regressed out','Rg normalized w/0.5 w/experimental pH regressed out', 'Rg normalized w/0.418 w/experimental pH regressed out',
      'Rg w/experimental buffer regressed out', 'Rg normalized w/0.427 w/experimental buffer regressed out','Rg normalized w/0.5 w/experimental buffer regressed out', 'Rg normalized w/0.418 w/experimental buffer regressed out']

for i in range(len(labelHeaders[1:])):
    labels.append([])
    inlierLabels.append([])

with open('training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        c = 0
        if counter > 0:
            for i in row[1:]:
                labels[c].append(i)
                c += 1
        counter += 1
with open('training/inliers.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        c = 0
        if counter > 0:
            for i in row[1:]:
                inlierLabels[c].append(i)
                c += 1
        counter += 1
inlierLabels = np.array(inlierLabels, dtype=float)
#print(np.shape(inlierLabels))

inlierFeatures = {}
outlierIndices = outlierIndices =  [123, 136, 151, 158, 171, 185]
inl = np.ones(190, dtype=bool)
inl[outlierIndices] = False

for l in layers6:
    inlierFeatures[l] = {}
    for p in PCAvals:
        x = np.load(f'6esmPCA/layer{l}_pca{p}.npy', allow_pickle=True)
        x = x[inl]
        #print(np.shape(x))
        inlierFeatures[l][p] = x


inlierFeatures12 = {}
outlierIndices = outlierIndices =  [123, 136, 151, 158, 171, 185]
inl = np.ones(190, dtype=bool)
inl[outlierIndices] = False

for l in layers12:
    inlierFeatures12[l] = {}
    for p in PCAvals:
        x = np.load(f'12esmPCA/layer{l}_pca{p}.npy', allow_pickle=True)
        x = x[inl]
        #print(np.shape(x))
        inlierFeatures12[l][p] = x

#labelHeaders = ['Sequence', 'Rg (nm)', 'Rg normalized w/0.427','Rg normalized w/0.5 (nm)', 'Rg normalized w/0.418 (nm)', 
    #'Rg w/pH regressed out', 'Rg normalized w/0.427 w/pH regressed out','Rg normalized w/0.5 w/pH regressed out', 'Rg normalized w/0.418 w/pH regressed out',
     #'Rg w/buffer regressed out', 'Rg normalized w/0.427 w/buffer regressed out','Rg normalized w/0.5 w/buffer regressed out', 'Rg normalized w/0.418 w/buffer regressed out', 
     #'Rg w/experimental pH regressed out', 'Rg normalized w/0.427 w/experimental pH regressed out','Rg normalized w/0.5 w/experimental pH regressed out', 'Rg normalized w/0.418 w/experimental pH regressed out',
      #'Rg w/experimental buffer regressed out', 'Rg normalized w/0.427 w/experimental buffer regressed out','Rg normalized w/0.5 w/experimental buffer regressed out', 'Rg normalized w/0.418 w/experimental buffer regressed out']

labelSplits = [
                ['Rg w/no norm', 'No regr out'],
                ['Rg norm w/0.427', 'No regr out'],
                ['Rg norm w/0.5', 'No regr out'],
                ['Rg norm w/0.418', 'No regr out'],
                ['Rg w/no norm', 'pH regr out'],
                ['Rg norm w/0.427', 'pH regr out'],
                ['Rg norm w/0.5', 'pH regr out'],
                ['Rg norm w/0.418', 'pH regr out'],
                ['Rg w/no norm', 'buffer regr out'],
                ['Rg norm w/0.427', 'buffer regr out'],
                ['Rg norm w/0.5', 'buffer regr out'],
                ['Rg norm w/0.418', 'buffer regr out'],
                ['Rg w/no norm', 'expr pH only regr out'],
                ['Rg norm w/0.427', 'expr pH only regr out'],
                ['Rg norm w/0.5', 'expr pH only regr out'],
                ['Rg norm w/0.418', 'expr pH only regr out'],
                ['Rg w/no norm', 'expr buffer only regr out'],
                ['Rg norm w/0.427', 'expr buffer only regr out'],
                ['Rg norm w/0.5', 'expr buffer only regr out'],
                ['Rg norm w/0.418', 'expr buffer only regr out'],
]


#Try Four other seeds...

with open('esmLosses2.csv', 'w', newline='') as f:
    seed = 43
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'ESM Mode', 'Layer', 'Principal Components', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]
        for l in layers6:
            for d in PCAvals:
                X = features[l][d]
                Xi = inlierFeatures[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-6',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-6' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        for l in layers12:
            for d in PCAvals:
                X = features12[l][d]
                Xi = inlierFeatures12[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-12',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-12' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        labelCounter += 1

with open('esmLosses3.csv', 'w', newline='') as f:
    seed = 44
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'ESM Mode', 'Layer', 'Principal Components', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]
        for l in layers6:
            for d in PCAvals:
                X = features[l][d]
                Xi = inlierFeatures[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-6',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-6' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        for l in layers12:
            for d in PCAvals:
                X = features12[l][d]
                Xi = inlierFeatures12[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-12',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-12' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        labelCounter += 1
with open('esmLosses4.csv', 'w', newline='') as f:
    seed = 44
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'ESM Mode', 'Layer', 'Principal Components', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]
        for l in layers6:
            for d in PCAvals:
                X = features[l][d]
                Xi = inlierFeatures[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-6',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-6' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        for l in layers12:
            for d in PCAvals:
                X = features12[l][d]
                Xi = inlierFeatures12[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-12',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-12' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        labelCounter += 1

with open('esmLosses5.csv', 'w', newline='') as f:
    seed = 45
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'ESM Mode', 'Layer', 'Principal Components', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]
        for l in layers6:
            for d in PCAvals:
                X = features[l][d]
                Xi = inlierFeatures[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-6',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-6' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        for l in layers12:
            for d in PCAvals:
                X = features12[l][d]
                Xi = inlierFeatures12[l][d]
                losses = evaluate_models_rmse(X, label, seed)
                lossesi = evaluate_models_rmse(Xi, inlierLabel, seed)
                for i in range(len(losses)):
                    row = [ls[0], ls[1], 'All', 'ESM-12',f'{l}', f'{d}', losses[i][0], losses[i][1], losses[i][2], losses[i][3]]
                    rowi = [ls[0], ls[1], 'Inliers','ESM-12' ,f'{l}', f'{d}', lossesi[i][0], lossesi[i][1], lossesi[i][2], lossesi[i][3]]
                    writer.writerow(row)
                    writer.writerow(rowi)
        labelCounter += 1
