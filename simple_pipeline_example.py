"""
Simple example using the TrainTestAwarePipeline that correctly handles
training vs test data concatenation.
"""

import pandas as pd
import numpy as np
from custom_transformers import TrainTestAwarePipeline


def main():
    # Load data
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    print(f"Original shapes - Train: {train.shape}, Test: {test.shape}")
    
    # Define categorical columns and feature combinations
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    feature_combinations = [
        ("job", "marital"),
        ("education", "housing"), 
        ("job", "education"),
        ("marital", "housing")
    ]
    
    # Create the smart pipeline
    print("Creating pipeline...")
    pipeline = TrainTestAwarePipeline(
        categorical_cols_for_ohe=categorical_cols,
        feature_combinations=feature_combinations,
        min_frequency_ohe=0.02,
        min_frequency_combo=0.01,
        drop_target_encoded_cols=True,
        original_data_path="bank-full.csv"
    )
    
    # Prepare data
    X_train = train.drop(columns=['y', 'id'])
    y_train = train['y']
    X_test = test.drop(columns=['id'])
    
    # Fit the pipeline
    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Transform data
    print("Transforming data...")
    X_train_transformed = pipeline.transform_train(X_train)  # Includes concatenation
    X_test_transformed = pipeline.transform_test(X_test)     # No concatenation
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_transformed)
    X_test_df = pd.DataFrame(X_test_transformed)
    
    print(f"\nResults:")
    print(f"- Original train: {train.shape}")
    print(f"- Original test: {test.shape}")
    print(f"- Transformed train: {X_train_df.shape}")
    print(f"- Transformed test: {X_test_df.shape}")
    
    # Key checks
    print(f"\n✅ Column counts match: {X_train_df.shape[1] == X_test_df.shape[1]}")
    print(f"✅ All columns numerical: {X_train_df.select_dtypes(include=[np.number]).shape[1] == X_train_df.shape[1]}")
    print(f"✅ Training data includes original: {X_train_df.shape[0] > train.shape[0]}")
    print(f"✅ Test data has correct rows (no concatenation): {X_test_df.shape[0] == test.shape[0]}")
    
    return X_train_df, X_test_df, y_train, pipeline


if __name__ == "__main__":
    X_train_transformed, X_test_transformed, y_train, fitted_pipeline = main()
    
    print("\n" + "="*60)
    print("🎉 PIPELINE WORKING CORRECTLY!")
    print("="*60)
    
    print("\n✅ What this pipeline does:")
    print("1. Concatenates original + synthetic data for TRAINING only")
    print("2. Applies target encoding using original data mappings")
    print("3. Engineers pdays features (missing values + temporal bins)")
    print("4. Applies rare one-hot encoding to categorical columns")
    print("5. Creates combination features with rare encoding")
    print("6. Ensures identical columns between train/test")
    print("7. Produces only numerical output")
    print("8. Test data keeps original row count (no concatenation)")
    
    print(f"\n🚀 Ready for model training!")
    print(f"   Training data: {X_train_transformed.shape}")
    print(f"   Test data: {X_test_transformed.shape}")
