import os
import pandas as pd
import numpy as np
from doAllRegressions import evaluate_models_rmse

def main(embedding_filename, target_column=None, output_dir=".", output_filename="regression_results.csv"):

    # 1. Construct full path from 'data' directory
    data_dir = "data"
    embedding_path = os.path.join(data_dir, embedding_filename)

    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found at: {embedding_path}")

    # 2. Load embeddings dynamically
    df = pd.read_csv(embedding_path)
    print(f"Loaded embeddings from {embedding_path}: {df.shape[0]} samples, {df.shape[1]} columns")

    # 3. Identify feature and target columns
    if target_column is None:
        target_column = df.columns[-1]
        print(f"No target column specified. Using last column: {target_column}")

    # 4. Extract and convert features/target to numeric arrays
    X = df.drop(columns=[target_column]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).values

    # 5. Run regression evaluation
    print("\nRunning regression evaluations...")
    results = evaluate_models_rmse(X, y)

    # 6. Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 7. Convert to DataFrame and save
    results_df = pd.DataFrame(results, columns=["Model", "Split", "R2", "RMSE"])
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    main(
        embedding_filename="protbertEmbeddingsPCA190.csv",
        output_dir="protBertRegressionResults",
        output_filename="PCA190RegressionSummary.csv"
    )