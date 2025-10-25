import csv 
import numpy as np
import matplotlib.pyplot as plt

lossesWithLinear = []
lossesWithRidge = []
lossesWithLasso = []
lossesWithKRR = []
lossesWithGPR = []

lossesOnAll = []
lossesOnInliers = []

lossesOnNoReg = []
lossesOnlowReg = []
lossesOnMidReg = []
lossesOnHighReg = []

bigSplit = []
medSplit = []
smallSplit = []

allLosses = []
with open('esmLosses.csv', newline='') as f:


    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter > 0:
            allLosses.append(row) 
            if row[6] == 'Linear':
                lossesWithLinear.append(row[-1])
            elif row[6] == 'Ridge':
                lossesWithRidge.append(row[-1])
            elif row[6] == 'Lasso':
                lossesWithLasso.append(row[-1])
            elif row[6] == 'Kernel Ridge':
                lossesWithKRR.append(row[-1])
            elif row[6] == 'GPR':
                lossesWithGPR.append(row[-1])
            
            if row[2] == 'All' and not row[6] == 'Linear':
                lossesOnAll.append(row[-1])
            elif not row[6] == 'Linear':
                lossesOnInliers.append(row[-1])

            if 'w/no' in row[0] and not row[6] == 'Linear':
                lossesOnNoReg.append(row[-1])
            elif '0.418' in row[0] and not row[6] == 'Linear':
                lossesOnlowReg.append(row[-1])
            elif '0.427' in row[0] and not row[6] == 'Linear':
                lossesOnMidReg.append(row[-1])
            elif '0.5' in row[0] and not row[6] == 'Linear':
                lossesOnHighReg.append(row[-1])
            
            if row[7] == '80/20' and not row[6] == 'Linear':
                bigSplit.append(row[-1])
            elif row[7] == '85/15' and not row[6] == 'Linear':
                medSplit.append(row[-1])
            elif row[7] == '90/10' and not row[6] == 'Linear':
                smallSplit.append(row[-1])
       
            
        counter += 1
print(len(allLosses))
print(len(lossesOnNoReg))  
allLosses = np.array(allLosses)
allLosses[:, -1] = allLosses[:, -1].astype(float)
sort_indices = allLosses[:, -1].argsort()
sorted = allLosses[sort_indices]

models = ["Linear", "Ridge", "Lasso", "Kernel Ridge", "GPR"]
losses = [np.mean(np.array(lossesWithLinear, dtype=float)), np.mean(np.array(lossesWithRidge, dtype=float)), np.mean(np.array(lossesWithLasso, dtype=float)), np.mean(np.array(lossesWithKRR, dtype=float)), np.mean(np.array(lossesWithGPR, dtype=float))]


plt.bar(models, losses)
plt.xlabel('Model Type')
plt.ylabel("Mean RSME Loss")

plt.show()


plt.bar(models[1:], losses[1:])
plt.xlabel('Model Type')
plt.ylabel("Mean RSME Loss")

plt.show()

points = ["All", "Inliers"]
losses = [np.mean(np.array(lossesOnAll, dtype=float)), np.mean(np.array(lossesOnInliers, dtype=float))]
plt.bar(points, losses)
plt.xlabel('Points Chosen')
plt.ylabel("Mean RSME Loss")

plt.show()

regs = ['No reg', 'Reg w/0.418', 'Reg w/0.427', 'Reg w/0.5']
losses = [np.mean(np.array(lossesOnNoReg, dtype=float)), np.mean(np.array(lossesOnlowReg, dtype=float)), np.mean(np.array(lossesOnMidReg, dtype=float)), np.mean(np.array(lossesOnHighReg, dtype=float))]
plt.bar(regs, losses)
plt.xlabel('Regularizations')
plt.ylabel("Mean RSME Loss")

plt.show()

splits = ['80/20', '85/15', '90/10']
losses = [np.mean(np.array(bigSplit, dtype=float)), np.mean(np.array(medSplit, dtype=float)), np.mean(np.array(smallSplit, dtype=float))]
plt.bar(splits, losses)
plt.xlabel('Splits')
plt.ylabel("Mean RSME Loss")

plt.show()

print(sorted[:20, :])

