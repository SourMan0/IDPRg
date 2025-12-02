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

features1 = []

with open('protein_features3St.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            features1.append(row[1:])
        counter += 1
features1 = np.array(features1, dtype='float')

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

with open('../training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        c = 0
        if counter > 0:
            for i in row[1:]:
                labels[c].append(i)
                c += 1
        counter += 1
with open('../training/inliers.csv', newline='') as f:
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

outlierIndices =  [123, 136, 151, 158, 171, 185]
inl = np.ones(190, dtype=bool)
inl[outlierIndices] = False

inlierFeatures1 = features1[inl]

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
print(np.shape(features1))


with open('pfeatureLosses1St3.csv', 'w', newline='') as f:
    seed = 42
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'Feature Selection', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]

        losses1 = evaluate_models_rmse(features1, label, seed)
        lossesi1 = evaluate_models_rmse(inlierFeatures1, inlierLabel, seed)
        for i in range(len(losses1)):
            row = [ls[0], ls[1], 'All', '1', losses1[i][0], losses1[i][1], losses1[i][2], losses1[i][3]]
            rowi = [ls[0], ls[1], 'Inliers','1', lossesi1[i][0], lossesi1[i][1], lossesi1[i][2], lossesi1[i][3]]
            writer.writerow(row)
            writer.writerow(rowi)
        
        labelCounter += 1
with open('pfeatureLosses2St3.csv', 'w', newline='') as f:
    seed = 43
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'Feature Selection', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]

        losses1 = evaluate_models_rmse(features1, label, seed)
        lossesi1 = evaluate_models_rmse(inlierFeatures1, inlierLabel, seed)
        for i in range(len(losses1)):
            row = [ls[0], ls[1], 'All', '1', losses1[i][0], losses1[i][1], losses1[i][2], losses1[i][3]]
            rowi = [ls[0], ls[1], 'Inliers','1', lossesi1[i][0], lossesi1[i][1], lossesi1[i][2], lossesi1[i][3]]
            writer.writerow(row)
            writer.writerow(rowi)
        
        labelCounter += 1

with open('pfeatureLosses3St3.csv', 'w', newline='') as f:
    seed = 44
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'Feature Selection', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]

        losses1 = evaluate_models_rmse(features1, label, seed)
        lossesi1 = evaluate_models_rmse(inlierFeatures1, inlierLabel, seed)
        for i in range(len(losses1)):
            row = [ls[0], ls[1], 'All', '1', losses1[i][0], losses1[i][1], losses1[i][2], losses1[i][3]]
            rowi = [ls[0], ls[1], 'Inliers','1', lossesi1[i][0], lossesi1[i][1], lossesi1[i][2], lossesi1[i][3]]
            writer.writerow(row)
            writer.writerow(rowi)
        
        labelCounter += 1

with open('pfeatureLosses4St3.csv', 'w', newline='') as f:
    seed = 45
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'Feature Selection', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]

        losses1 = evaluate_models_rmse(features1, label, seed)
        lossesi1 = evaluate_models_rmse(inlierFeatures1, inlierLabel, seed)
        for i in range(len(losses1)):
            row = [ls[0], ls[1], 'All', '1', losses1[i][0], losses1[i][1], losses1[i][2], losses1[i][3]]
            rowi = [ls[0], ls[1], 'Inliers','1', lossesi1[i][0], lossesi1[i][1], lossesi1[i][2], lossesi1[i][3]]
            writer.writerow(row)
            writer.writerow(rowi)
        
        labelCounter += 1

with open('pfeatureLosses5St3.csv', 'w', newline='') as f:
    seed = 46
    writer = csv.writer(f)
    header = ['Normlaization', 'Regressing out', 'Points', 'Feature Selection', 'Regression Type', 'Test Split', 'Test R2 Score', 'RMSE Score']
    writer.writerow(header)
    labelCounter = 0
    for ls in labelSplits:
        label =  labels[labelCounter]
        inlierLabel = inlierLabels[labelCounter]

        losses1 = evaluate_models_rmse(features1, label, seed)
        lossesi1 = evaluate_models_rmse(inlierFeatures1, inlierLabel, seed)
        for i in range(len(losses1)):
            row = [ls[0], ls[1], 'All', '1', losses1[i][0], losses1[i][1], losses1[i][2], losses1[i][3]]
            rowi = [ls[0], ls[1], 'Inliers','1', lossesi1[i][0], lossesi1[i][1], lossesi1[i][2], lossesi1[i][3]]
            writer.writerow(row)
            writer.writerow(rowi)
        labelCounter += 1
    