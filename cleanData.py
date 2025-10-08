#creating a dataset of just the sequences and corresponding Rg / length
import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from writeCsv import write_csv

sequences = []
Rgs = []
lengths = []
with open('data/rawData.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] != '' and row[3] != '':
            sequences.append(row[2])
            Rgs.append(row[3])
            lengths.append(len(row[2]))
#print(Rgs)

sequences = sequences[1:]
Rgs = Rgs[1:]
lengths = lengths[1:]
lengths = np.array(lengths)
sequences = np.array(sequences)
Rgs = np.array(Rgs, dtype = float)
#Finding the correct coeffecient
logN = np.log(lengths).reshape(-1, 1)
logRg = np.log(Rgs)

model = LinearRegression().fit(logN, logRg)
nu = model.coef_[0]
a = np.exp(model.intercept_)
print("Fitted ν =", nu)

#rebuild dataset:
plt.figure(figsize=(6, 5))

# Scatter points
plt.scatter(lengths, Rgs, s=25, alpha=0.7, label='data')

# Fitted line
N_fit = np.linspace(min(lengths), max(lengths), 100)
Rg_fit = a * (N_fit ** nu)
plt.plot(N_fit, Rg_fit, 'r-', lw=2, label=f'Fit: Rg = {a:.2f} * N^{nu:.3f}')

# Log–log scale
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Sequence length (N)')
plt.ylabel('Radius of gyration (nm)')
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.show()
plt.close()


#Compute predictions + residuals

logRg_pred = model.predict(logN)
residuals = logRg - logRg_pred

#look for outliers
sigma = np.std(residuals)
outlier_mask = np.abs(residuals) > 2 * sigma
outlier_indices = np.where(outlier_mask)[0]
print("Outlier indices:", outlier_indices)
outlierRgs = []
outlierNs = []
for i in outlier_indices:
    print(f"N={lengths[i]}, Rg={Rgs[i]}, residual={residuals[i]:.3f}")
    outlierNs.append(lengths[i])
    outlierRgs.append(Rgs[i])

#plot

# All data
plt.scatter(lengths, Rgs, alpha=0.6, label='Data')

# Outliers
plt.scatter(outlierNs, outlierRgs, color='red', s=60, label='Outliers')

# Fitted line
N_fit = np.linspace(min(lengths), max(lengths), 200)
Rg_fit = a * (N_fit ** nu)
plt.plot(N_fit, Rg_fit, 'k--', lw=2, label=f'Fit: Rg = {a:.2f} * N^{nu:.3f}')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sequence length (N)')
plt.ylabel('Radius of gyration (Rg)')
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.show()



#refit without outliers
mask_inliners = ~outlier_mask
logNIn = logN[mask_inliners]
logRgIn = logRg[mask_inliners]

modelRefit = LinearRegression().fit(logNIn, logRgIn)
nuRefit = modelRefit.coef_[0]
logaRefit = modelRefit.intercept_
aRefit = np.exp(logaRefit)

plt.figure(figsize=(6,5))

# Points
plt.scatter(lengths[mask_inliners], Rgs[mask_inliners],
            color='gray', alpha=0.7, label='Inliers')
plt.scatter(lengths[outlier_mask], Rgs[outlier_mask],
            color='red', s=60, label='Outliers')

# Fit lines
N_fit = np.linspace(lengths.min(), lengths.max(), 200)
Rg_fit_init  = a  * (N_fit ** nu)
Rg_fit_refit = aRefit * (N_fit ** nuRefit)

plt.plot(N_fit, Rg_fit_init,  'k--', lw=2,
         label=f'Initial fit:  ν={nu:.3f}')
plt.plot(N_fit, Rg_fit_refit, 'b-',  lw=2,
         label=f'Refit w/o outliers:  ν={nuRefit:.3f}')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sequence length (N)')
plt.ylabel('Radius of gyration (Rg)')
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.show()

RgNorminit = Rgs / (lengths ** nu)
RgNormRefit = Rgs / (lengths ** nuRefit)
#Try standard value of v = 0.5
RgNormSt = Rgs / (lengths ** 0.5)


#Uncomment if you want to remake the files
'''
# 1. All data (raw)
write_csv('allRaw.csv', sequences, Rgs, "Protein Sequence", "Rg (nm)")

# 2. All data (normalized with ν = 0.427)
write_csv('allNormalized.csv', sequences, RgNorminit, "Protein Sequence", "Normalized Rg with ν = 0.427 (nm)")

# 3. Inliers only (raw)
write_csv('inliersRaw.csv', sequences[mask_inliners], Rgs[mask_inliners], "Protein Sequence", "Rg (nm)")

# 4. Inliers only (normalized with ν = 0.418)
write_csv('inliersNormalized.csv', sequences[mask_inliners], RgNormRefit[mask_inliners], "Protein Sequence", "Normalized Rg with ν = 0.418  (nm)")

#5 All data (normalized at ν = 0.5)
write_csv('allNormalizedNaive.csv', sequences, RgNormSt, "Protein Sequence", "Normalized Rg with ν = 0.5 (nm)")

#6 All data (normalized  ν = 0.418)
write_csv('allNormalizedWithInliers.csv', sequences, RgNormRefit, "Protein Sequence", "Normalized Rg with ν = 0.418 (nm)")

#7 Inlier data (normalized  ν = 0.427)
write_csv('inliersNormalizedWithAll.csv', sequences[mask_inliners], RgNorminit[mask_inliners], "Protein Sequence", "Normalized Rg with ν = 0.427 (nm)")

#9 Inlier data (normalized at ν = 0.5)
write_csv('inliersNormalizedNaive.csv', sequences[mask_inliners], RgNormSt[mask_inliners], "Protein Sequence", "Normalized Rg with ν = 0.5 (nm)")
'''