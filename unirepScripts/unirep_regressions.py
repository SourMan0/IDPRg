import unirepEmbeddings
import pcaCalc as pca
import doAllRegressions as reg
from pathlib import Path
import pandas as pd
import ast
import os


for random_state in range(1,6):
    # Create embeddings if they do not exist
    paths = ["data/unirep_allRaw.csv", "data/unirep_inliersRaw.csv"]
    components_list = [10, 20, 50, 100, 150, 167]
    emb_col="UniRep Embedding"
    rg_mappings = {"Rg (nm)": ("Rg w/no norm", "No regr out"),"Rg normalized w/0.421": ("Rg norm w/0.421", "No regr out"),"Rg normalized w/0.5 (nm)": ("Rg norm w/0.5", "No regr out"),"Rg normalized w/0.406 (nm)": ("Rg norm w/0.406", "No regr out"),"Rg w/pH regressed out": ("Rg w/no norm", "pH regr out"),"Rg normalized w/0.421 w/pH regressed out": ("Rg norm w/0.421", "pH regr out"),"Rg normalized w/0.5 w/pH regressed out": ("Rg norm w/0.5", "pH regr out"),"Rg normalized w/0.406 w/pH regressed out": ("Rg norm w/0.406", "pH regr out"),"Rg w/buffer regressed out": ("Rg w/no norm", "buffer regr out"),"Rg normalized w/0.421 w/buffer regressed out": ("Rg norm w/0.421", "buffer regr out"),"Rg normalized w/0.5 w/buffer regressed out": ("Rg norm w/0.5", "buffer regr out"),"Rg normalized w/0.406 w/buffer regressed out": ("Rg norm w/0.406", "buffer regr out"),"Rg w/experimental pH regressed out": ("Rg w/no norm", "expr pH only regr out"),"Rg normalized w/0.421 w/experimental pH regressed out": ("Rg norm w/0.421", "expr pH only regr out"),"Rg normalized w/0.5 w/experimental pH regressed out": ("Rg norm w/0.5", "expr pH only regr out"),"Rg normalized w/0.406 w/experimental pH regressed out": ("Rg norm w/0.406", "expr pH only regr out"),"Rg w/experimental buffer regressed out": ("Rg w/no norm", "expr buffer only regr out"),"Rg normalized w/0.421 w/experimental buffer regressed out": ("Rg norm w/0.421", "expr buffer only regr out"),"Rg normalized w/0.5 w/experimental buffer regressed out": ("Rg norm w/0.5", "expr buffer only regr out"),"Rg normalized w/0.406 w/experimental buffer regressed out": ("Rg norm w/0.406", "expr buffer only regr out")}


    for path in paths:
        if not Path(path).is_file():
            print(f"Creating UniRep embeddings for {path}...")
            unirepEmbeddings.unirep_embed(path, pca_toggle=False, pca_num_components=190)

    # Run PCA
    all_paths = ["data/unirep_allRaw.csv"]
    for n_components in components_list:
        if not Path(f"data/{Path('data/unirep_allRaw.csv').stem}PCA{n_components}.csv").is_file():
            print(f"Creating PCA for {"data/unirep_allRaw.csv"} with {n_components} components...")
            pca.pca_calc("data/unirep_allRaw.csv", n_components, emb_col=emb_col)
        all_paths.append(f"data/{Path('data/unirep_allRaw.csv').stem}PCA{n_components}.csv")
    print(all_paths)


    inlier_paths = ["data/unirep_inliersRaw.csv"]
    for n_components in components_list:
        components_check = n_components
        if n_components == 167:
            components_check = 163
        if not Path(f"data/{Path('data/unirep_inliersRaw.csv').stem}PCA{components_check}.csv").is_file():
            print(f"Creating PCA for {"data/unirep_inliersRaw.csv"} with {components_check} components...")
            pca.pca_calc("data/unirep_inliersRaw.csv", components_check, emb_col=emb_col)
        inlier_paths.append(f"data/{Path('data/unirep_inliersRaw.csv').stem}PCA{components_check}.csv")
    print("Starting regressions...")

    components_list.insert(0, -1) # For no PCA case

    # results = pd.DataFrame(columns=["Normalization","Regressing out","Points","Principal Components","Regression Type","Test Split","Test R2 Score","RMSE Score"])

    # all_points = pd.read_csv("training/all_points.csv")
    # inliers = pd.read_csv("training/inliers.csv")

    # total = len(all_paths) * len(rg_mappings) + len(inlier_paths) * len(rg_mappings)
    # count = 0

    # for i in range(len(all_paths)):
    #     for rg_col, (norm_type, regr_type) in rg_mappings.items():
    #         pca = components_list[i]
    #         df_X = pd.read_csv(all_paths[i], converters={emb_col: ast.literal_eval})
    #         X = df_X[emb_col].tolist()
    #         y = all_points[rg_col].tolist()
    #         single_results = reg.evaluate_models_rmse(X, y)
    #         for result in single_results:
    #             results.loc[len(results)] = {"Normalization": norm_type, "Regressing out": regr_type, "Points": "All", "Principal Components": pca, "Regression Type": result[0], "Test Split": result[1], "Test R2 Score": result[2], "RMSE Score": result[3]}
    #         count += 1
    #         if count % 5 == 0:
    #             print(f"Completed {count}/{total} sets of regressions.")

    # for i in range(len(inlier_paths)):
    #     for rg_col, (norm_type, regr_type) in rg_mappings.items():
    #         pca = components_list[i]
    #         df_X = pd.read_csv(inlier_paths[i], converters={emb_col: ast.literal_eval})
    #         X = df_X[emb_col].tolist()
    #         y = inliers[rg_col].tolist()
    #         single_results = reg.evaluate_models_rmse(X, y)
    #         for result in single_results:
    #             results.loc[len(results)] = {"Normalization": norm_type, "Regressing out": regr_type, "Points": "Inliers", "Principal Components": pca, "Regression Type": result[0], "Test Split": result[1], "Test R2 Score": result[2], "RMSE Score": result[3]}
    #         count += 1
    #         if count % 5 == 0:
    #             print(f"Completed {count}/{total} sets of regressions.")

    # results.to_csv("unirepLosses.csv", index=False)


    results = pd.DataFrame(columns=["Normalization","Regressing out","Points","Principal Components","Regression Type","Test Split","Test R2 Score","RMSE Score"])

    all_points = pd.read_csv("training/all_points.csv")
    inliers = pd.read_csv("training/inliers.csv")

    output_csv = f"losses/unirepLosses{random_state}.csv"
    file_exists = os.path.isfile(output_csv)

    # if file doesn't exist yet, write header once so the CSV is initialized
    if not file_exists:
        results.to_csv(output_csv, index=False)

    total = len(all_paths) * len(rg_mappings) + len(inlier_paths) * len(rg_mappings)
    count = 0

    for i in range(len(all_paths)):
        for rg_col, (norm_type, regr_type) in rg_mappings.items():
            pca = components_list[i]
            df_X = pd.read_csv(all_paths[i], converters={emb_col: ast.literal_eval})
            X = df_X[emb_col].tolist()
            y = all_points[rg_col].tolist()
            single_results = reg.evaluate_models_rmse(X, y, random_state=random_state)
            for result in single_results:
                results.loc[len(results)] = {
                    "Normalization": norm_type,
                    "Regressing out": regr_type,
                    "Points": "All",
                    "Principal Components": pca,
                    "Regression Type": result[0],
                    "Test Split": result[1],
                    "Test R2 Score": result[2],
                    "RMSE Score": result[3]
                }
                # append just this last row to csv (no header)
                results.tail(1).to_csv(output_csv, mode='a', header=False, index=False)

            count += 1
            if count % 5 == 0:
                print(f"Completed {count}/{total} sets of regressions.")

    for i in range(len(inlier_paths)):
        for rg_col, (norm_type, regr_type) in rg_mappings.items():
            pca = components_list[i]
            df_X = pd.read_csv(inlier_paths[i], converters={emb_col: ast.literal_eval})
            X = df_X[emb_col].tolist()
            y = inliers[rg_col].tolist()
            single_results = reg.evaluate_models_rmse(X, y, random_state=random_state)
            for result in single_results:
                results.loc[len(results)] = {
                    "Normalization": norm_type,
                    "Regressing out": regr_type,
                    "Points": "Inliers",
                    "Principal Components": pca,
                    "Regression Type": result[0],
                    "Test Split": result[1],
                    "Test R2 Score": result[2],
                    "RMSE Score": result[3]
                }
                # append just this last row to csv (no header)
                results.tail(1).to_csv(output_csv, mode='a', header=False, index=False)

            count += 1
            if count % 5 == 0:
                print(f"Completed {count}/{total} sets of regressions.")