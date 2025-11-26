import unirepEmbeddings

for i in ["data/allRaw.csv", "data/inliersRaw.csv", "data/allNormalized.csv", "data/allNormalizedWithInliers.csv", "data/allNormalizedNaive.csv", "data/inliersNormalized.csv", "data/inliersNormalizedWithAll.csv", "data/inliersNormalizedNaive.csv"]:
    unirepEmbeddings.unirep_embed(i, pca_toggle=False, pca_num_components=190)