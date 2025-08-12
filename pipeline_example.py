"""
Example usage of the custom transformers pipeline.

This script demonstrates how to use the complete preprocessing pipeline
that implements your data processing steps.
"""

import pandas as pd
import numpy as np
from custom_transformers import create_single_pipeline, create_test_pipeline_from_fitted


def main():
    # Load your data
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    print(f"Original shapes - Train: {train.shape}, Test: {test.shape}")
    
    # Define which categorical columns to apply rare one-hot encoding to
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    # Define feature combinations to create
    feature_combinations = [
        ("job", "marital"),       # Job + marital status combinations
        ("education", "housing"), # Education + housing combinations  
        ("job", "education"),     # Job + education combinations
        ("marital", "housing"),   # Marital + housing combinations
        ("job", "contact"),       # Job + contact method combinations
    ]
    
    # Create the training pipeline (includes data concatenation)
    print("Creating training pipeline...")
    train_pipeline = create_single_pipeline(
        categorical_cols_for_ohe=categorical_cols,
        feature_combinations=feature_combinations,
        min_frequency_ohe=0.02,         # Categories with <2% frequency become 'rare'
        min_frequency_combo=0.01,       # Combinations with <1% frequency become 'rare'
        drop_target_encoded_cols=True,  # Drop original categorical columns after target encoding
        original_data_path="bank-full.csv",
        for_training=True  # This includes data concatenation
    )
    
    # Prepare training data (exclude target and id columns)
    X_train = train.drop(columns=['y', 'id'])
    y_train = train['y']
    
    # Prepare test data (exclude id column)
    X_test = test.drop(columns=['id'])
    
    # Fit the pipeline on training data
    print("Fitting training pipeline...")
    train_pipeline.fit(X_train, y_train)
    
    # Transform training data
    print("Transforming training data...")
    X_train_transformed = train_pipeline.transform(X_train)
    
    # Create test pipeline that shares fitted transformers but doesn't concatenate original data
    print("Creating test pipeline...")
    test_pipeline = create_test_pipeline_from_fitted(train_pipeline)
    
    # Transform test data
    print("Transforming test data...")
    X_test_transformed = test_pipeline.transform(X_test)
    
    # Convert to DataFrames for easier handling
    X_train_df = pd.DataFrame(X_train_transformed)
    X_test_df = pd.DataFrame(X_test_transformed)
    
    print(f"Transformed shapes - Train: {X_train_df.shape}, Test: {X_test_df.shape}")
    print(f"Column count matches: {X_train_df.shape[1] == X_test_df.shape[1]}")
    
    # Verify all columns are numerical
    train_all_numerical = X_train_df.select_dtypes(include=[np.number]).shape[1] == X_train_df.shape[1]
    test_all_numerical = X_test_df.select_dtypes(include=[np.number]).shape[1] == X_test_df.shape[1]
    
    print(f"All training columns are numerical: {train_all_numerical}")
    print(f"All test columns are numerical: {test_all_numerical}")
    
    # IMPORTANT: Verify test data doesn't have extra rows from original data
    print(f"Test data has correct number of rows (no concatenation): {X_test_df.shape[0] == test.shape[0]}")
    print(f"Training data includes original data: {X_train_df.shape[0] > train.shape[0]}")
    
    # Show some information about the transformed data
    print(f"\nFeature engineering summary:")
    print(f"- Original categorical columns: {len(categorical_cols)}")
    print(f"- Feature combinations created: {len(feature_combinations)}")
    print(f"- Final feature count: {X_train_df.shape[1]}")
    
    # Show column names to understand what was created
    print(f"\nSample of created columns:")
    print(X_train_df.columns.tolist()[:20])  # Show first 20 columns
    
    return X_train_df, X_test_df, y_train, train_pipeline, test_pipeline


def create_minimal_pipeline():
    """
    Creates a minimal pipeline with just the essential transformations.
    Returns training pipeline only.
    """
    # Minimal categorical columns
    categorical_cols = ['job', 'marital', 'education', 'housing']
    
    # Minimal feature combinations
    feature_combinations = [
        ("job", "marital"),
        ("education", "housing")
    ]
    
    pipeline = create_single_pipeline(
        categorical_cols_for_ohe=categorical_cols,
        feature_combinations=feature_combinations,
        min_frequency_ohe=0.05,        # Higher threshold for fewer features
        min_frequency_combo=0.02,      # Higher threshold for fewer combinations
        drop_target_encoded_cols=True, # Clean up by dropping original columns
        original_data_path="bank-full.csv",
        for_training=True
    )
    
    return pipeline


def create_comprehensive_pipeline():
    """
    Creates a comprehensive pipeline with many feature combinations.
    Returns training pipeline only.
    """
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    # More extensive feature combinations
    feature_combinations = [
        ("job", "marital"),
        ("job", "education"), 
        ("job", "housing"),
        ("marital", "education"),
        ("marital", "housing"),
        ("education", "housing"),
        ("job", "contact"),
        ("marital", "contact"),
        ("housing", "loan"),
        ("education", "default"),
    ]
    
    pipeline = create_single_pipeline(
        categorical_cols_for_ohe=categorical_cols,
        feature_combinations=feature_combinations,
        min_frequency_ohe=0.01,        # Lower threshold for more features
        min_frequency_combo=0.005,     # Lower threshold for more combinations
        drop_target_encoded_cols=True,
        original_data_path="bank-full.csv",
        for_training=True
    )
    
    return pipeline


if __name__ == "__main__":
    # Run the main example
    X_train_transformed, X_test_transformed, y_train, train_pipeline, test_pipeline = main()
    
    print("\n" + "="*50)
    print("PIPELINE IMPLEMENTATION COMPLETE!")
    print("="*50)
    
    print("\nWhat this pipeline does:")
    print("1. ✅ Concatenates original + synthetic data (TRAINING ONLY)")
    print("2. ✅ Applies target encoding to categorical columns")
    print("3. ✅ Engineers pdays features (missing value handling + temporal bins)")
    print("4. ✅ Applies rare one-hot encoding to categorical columns") 
    print("5. ✅ Creates combination features with rare encoding")
    print("6. ✅ Ensures consistent columns between train/test")
    print("7. ✅ Produces only numerical columns in output")
    print("8. ✅ Test data does NOT get original data concatenated")
    
    print(f"\nYour transformed data is ready to use:")
    print(f"- Training data shape: {X_train_transformed.shape}")
    print(f"- Test data shape: {X_test_transformed.shape}")
    print("You can now train any model on X_train_transformed and y_train!")
