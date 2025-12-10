import csv 
import numpy as np
import matplotlib.pyplot as plt

#added cycling over all of the seeds

seeds = ['1', '2', '3', '4', '5']
bestFeatures = []

for i in seeds:

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


    # NEW: track losses by principal components
    pc_losses = {}

    allLosses = []
    with open(f'losses/unirepLosses{i}.csv', newline='') as f:
        reader = csv.reader(f)
        counter = 0
        for row in reader:
            if counter > 0:
                allLosses.append([i] + row)

                # row[4] = Regression Type
                if row[4] == 'Linear':
                    lossesWithLinear.append(row[-1])
                elif row[4] == 'Ridge':
                    lossesWithRidge.append(row[-1])
                elif row[4] == 'Lasso':
                    lossesWithLasso.append(row[-1])
                elif row[4] == 'Kernel Ridge':
                    lossesWithKRR.append(row[-1])
                elif row[4] == 'GPR':
                    lossesWithGPR.append(row[-1])
                
                # row[2] = Points ("All" or "Inliers")
                # exclude Linear
                if row[2] == 'All' and row[4] != 'Linear':
                    lossesOnAll.append(row[-1])
                elif row[4] != 'Linear':
                    lossesOnInliers.append(row[-1])

                # row[0] = Normalization (regularization condition text)
                # exclude Linear
                if 'w/no' in row[0] and row[4] != 'Linear':
                    lossesOnNoReg.append(row[-1])
                elif '0.418' in row[0] and row[4] != 'Linear':
                    lossesOnlowReg.append(row[-1])
                elif '0.427' in row[0] and row[4] != 'Linear':
                    lossesOnMidReg.append(row[-1])
                elif '0.5' in row[0] and row[4] != 'Linear':
                    lossesOnHighReg.append(row[-1])
                
                # row[5] = Test Split (80/20 etc.)
                # exclude Linear
                if row[5] == '80/20' and row[4] != 'Linear':
                    bigSplit.append(row[-1])
                elif row[5] == '85/15' and row[4] != 'Linear':
                    medSplit.append(row[-1])
                elif row[5] == '90/10' and row[4] != 'Linear':
                    smallSplit.append(row[-1])

                # NEW: principal components mean loss tracking
                # row[3] = Principal Components, row[-1] = RMSE Score
                pc_key = row[3]
                if pc_key not in pc_losses:
                    pc_losses[pc_key] = []
                pc_losses[pc_key].append(row[-1])
        
            counter += 1

    print(len(allLosses))
    print(len(lossesOnNoReg))  

    allLosses = np.array(allLosses)
    allLosses[:, -1] = allLosses[:, -1].astype(float)
    sort_indices = allLosses[:, -1].argsort()
    sorted_rows = allLosses[sort_indices]
    sorted = allLosses[sort_indices]


    models = ["Linear", "Ridge", "Lasso", "Kernel Ridge", "GPR"]
    losses = [
        np.mean(np.array(lossesWithLinear, dtype=float)),
        np.mean(np.array(lossesWithRidge, dtype=float)),
        np.mean(np.array(lossesWithLasso, dtype=float)),
        np.mean(np.array(lossesWithKRR, dtype=float)),
        np.mean(np.array(lossesWithGPR, dtype=float))
    ]

    plt.bar(models, losses)
    plt.xlabel('Model Type')
    plt.ylabel("Mean RMSE Loss")
    plt.show()

    plt.bar(models[1:], losses[1:])
    plt.xlabel('Model Type')
    plt.ylabel("Mean RMSE Loss")
    plt.show()

    points = ["All", "Inliers"]
    losses = [
        np.mean(np.array(lossesOnAll, dtype=float)),
        np.mean(np.array(lossesOnInliers, dtype=float))
    ]
    plt.bar(points, losses)
    plt.xlabel('Points Chosen')
    plt.ylabel("Mean RMSE Loss")
    plt.show()

    regs = ['No reg', 'Reg w/0.418', 'Reg w/0.427', 'Reg w/0.5']
    losses = [
        np.mean(np.array(lossesOnNoReg, dtype=float)),
        np.mean(np.array(lossesOnlowReg, dtype=float)),
        np.mean(np.array(lossesOnMidReg, dtype=float)),
        np.mean(np.array(lossesOnHighReg, dtype=float))
    ]
    plt.bar(regs, losses)
    plt.xlabel('Regularizations')
    plt.ylabel("Mean RMSE Loss")
    plt.show()

    splits = ['80/20', '85/15', '90/10']
    losses = [
        np.mean(np.array(bigSplit, dtype=float)),
        np.mean(np.array(medSplit, dtype=float)),
        np.mean(np.array(smallSplit, dtype=float))
    ]
    plt.bar(splits, losses)
    plt.xlabel('Splits')
    plt.ylabel("Mean RMSE Loss")
    plt.show()

    # NEW: bar graph of Principal Components vs mean RMSE loss
    pc_labels = list(pc_losses.keys())
    pc_mean_losses = [
        np.mean(np.array(pc_losses[pc], dtype=float)) for pc in pc_labels
    ]
    plt.bar(pc_labels, pc_mean_losses)
    plt.xlabel('Principal Components')
    plt.ylabel('Mean RMSE Loss')
    plt.show()

    print(sorted_rows[:20, :])
    bestFeatures.extend(list(sorted[:20, :]))

bestFeatures = np.array(bestFeatures)
bestFeatures[:, -1] = bestFeatures[:, -1].astype(float)
sort_indices = bestFeatures[:, -1].argsort()
sorted = bestFeatures[sort_indices]
print("Overall best Ones")
print(sorted[:20, :])
