import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from itertools import combinations


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder that computes target means from original data and applies to full dataset.
    """
    def __init__(self, original_data_path="bank-full.csv", drop_original_cols=False):
        self.original_data_path = original_data_path
        self.drop_original_cols = drop_original_cols
        self.target_maps_ = {}
        self.categorical_cols_ = []
        
    def fit(self, X, y=None):
        # Load original data to compute target means
        original = pd.read_csv(self.original_data_path, sep=";")
        original['y'] = original['y'].apply(lambda x: 1 if x=="yes" else 0)
        
        # Convert day to string to match your pipeline
        original['day'] = original['day'].astype(str)
        
        # Get categorical columns
        self.categorical_cols_ = original.select_dtypes(include=['object']).columns.tolist()
        if 'y' in self.categorical_cols_:
            self.categorical_cols_.remove('y')
            
        # Compute target means from original data
        for col in self.categorical_cols_:
            self.target_maps_[col] = original.groupby(col)['y'].mean().to_dict()
            
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Convert day to string to match training
        if 'day' in X_transformed.columns:
            X_transformed['day'] = X_transformed['day'].astype(str)
        
        # Apply target encoding
        for col in self.categorical_cols_:
            if col in X_transformed.columns:
                X_transformed[col + "_mean"] = X_transformed[col].map(self.target_maps_[col])
                
                # Drop original categorical column if requested
                if self.drop_original_cols:
                    X_transformed = X_transformed.drop(columns=[col])
                    
        return X_transformed


class RareOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encoder that groups rare categories (below min_frequency) into a 'rare' category.
    """
    def __init__(self, min_frequency=0.01, handle_unknown='ignore'):
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self.encoders_ = {}
        self.feature_names_ = {}
        self.rare_categories_ = {}
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col in X_df.columns:
            # Calculate category frequencies
            value_counts = X_df[col].value_counts()
            total_count = len(X_df)
            frequencies = value_counts / total_count
            
            # Identify rare categories
            rare_cats = frequencies[frequencies < self.min_frequency].index.tolist()
            self.rare_categories_[col] = rare_cats
            
            # Create modified series with rare categories grouped
            X_modified = X_df[col].copy()
            X_modified = X_modified.replace(rare_cats, 'RARE_CATEGORY')
            
            # Fit OneHotEncoder on modified data
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown=self.handle_unknown)
            encoder.fit(X_modified.values.reshape(-1, 1))
            
            self.encoders_[col] = encoder
            
            # Store feature names
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]  # Skip first due to drop='first'
            self.feature_names_[col] = feature_names
            
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        all_features = []
        feature_names = []
        
        for col in X_df.columns:
            # Apply rare category grouping
            X_modified = X_df[col].copy()
            X_modified = X_modified.replace(self.rare_categories_[col], 'RARE_CATEGORY')
            
            # Transform using fitted encoder
            encoded = self.encoders_[col].transform(X_modified.values.reshape(-1, 1))
            all_features.append(encoded)
            feature_names.extend(self.feature_names_[col])
            
        # Concatenate all encoded features
        result = np.concatenate(all_features, axis=1)
        
        return pd.DataFrame(result, columns=feature_names, index=X_df.index)
    
    def get_feature_names_out(self, input_features=None):
        all_names = []
        for col in self.feature_names_:
            all_names.extend(self.feature_names_[col])
        return np.array(all_names)


class CombinationFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Creates combination features from categorical columns and applies rare one-hot encoding.
    """
    def __init__(self, feature_combinations, min_frequency=0.01):
        self.feature_combinations = feature_combinations  # List of tuples like [("job", "marital"), ("education", "housing")]
        self.min_frequency = min_frequency
        self.rare_encoder_ = None
        self.combination_names_ = []
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Create combination features
        combination_data = {}
        self.combination_names_ = []
        
        for combo in self.feature_combinations:
            combo_name = "_".join(combo)
            self.combination_names_.append(combo_name)
            
            # Create combination column
            combination_data[combo_name] = X_df[list(combo)].apply(
                lambda x: "_".join(x.astype(str)), axis=1
            )
        
        # Create DataFrame with combination features
        combo_df = pd.DataFrame(combination_data, index=X_df.index)
        
        # Fit rare encoder on combination features
        self.rare_encoder_ = RareOneHotEncoder(min_frequency=self.min_frequency)
        self.rare_encoder_.fit(combo_df)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Create combination features
        combination_data = {}
        
        for combo in self.feature_combinations:
            combo_name = "_".join(combo)
            combination_data[combo_name] = X_df[list(combo)].apply(
                lambda x: "_".join(x.astype(str)), axis=1
            )
        
        # Create DataFrame with combination features
        combo_df = pd.DataFrame(combination_data, index=X_df.index)
        
        # Transform using fitted rare encoder
        return self.rare_encoder_.transform(combo_df)
    
    def get_feature_names_out(self, input_features=None):
        return self.rare_encoder_.get_feature_names_out()


class PdaysFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for pdays column: handle missing values and create temporal bins.
    """
    def __init__(self, create_bins=True):
        self.create_bins = create_bins
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        if 'pdays' in X_df.columns:
            # Create indicator for missing pdays
            X_df['pdays_none'] = (X_df['pdays'] == -1).astype(int)
            
            # Replace -1 with NaN
            X_df['pdays'] = X_df['pdays'].replace(-1, np.nan)
            
            if self.create_bins:
                # Create temporal bins as in your pipeline
                X_df['pdays_qtr_yr'] = ((X_df['pdays'] > 84) & (X_df['pdays'] < 96)).astype(int)
                X_df['pdays_hlf_yr'] = ((X_df['pdays'] > 175) & (X_df['pdays'] < 190)).astype(int)
                X_df['pdays_fl_yr'] = ((X_df['pdays'] > 359) & (X_df['pdays'] < 372)).astype(int)
                X_df['pdays_mor_yr'] = (X_df['pdays'] > 371).astype(int)
            
        return X_df


class DataConcatenator(BaseEstimator, TransformerMixin):
    """
    Concatenates original data with the input data (synthetic) for training only.
    For test data, only applies the same preprocessing without adding rows.
    """
    def __init__(self, original_data_path="bank-full.csv", is_training=True):
        self.original_data_path = original_data_path
        self.is_training = is_training
        self.original_data_ = None
        
    def fit(self, X, y=None):
        # Load and prepare original data
        self.original_data_ = pd.read_csv(self.original_data_path, sep=";")
        self.original_data_['y'] = self.original_data_['y'].apply(lambda x: 1 if x=="yes" else 0)
        self.original_data_['day'] = self.original_data_['day'].astype(str)
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Convert day to string to match original
        if 'day' in X_df.columns:
            X_df['day'] = X_df['day'].astype(str)
            
        # Only concatenate with original data if this is training data
        if self.is_training:
            result = pd.concat([X_df, self.original_data_], ignore_index=True)
        else:
            result = X_df
            
        return result


def create_complete_pipeline(
    categorical_cols_for_ohe=None,
    feature_combinations=None,
    min_frequency_ohe=0.01,
    min_frequency_combo=0.01,
    drop_target_encoded_cols=False,
    original_data_path="bank-full.csv"
):
    """
    Creates a complete preprocessing pipeline that implements your data processing steps.
    
    Parameters:
    -----------
    categorical_cols_for_ohe : list
        Categorical columns to apply rare one-hot encoding to
    feature_combinations : list of tuples
        Feature combinations to create, e.g. [("job", "marital"), ("education", "housing")]
    min_frequency_ohe : float
        Minimum frequency threshold for rare one-hot encoding
    min_frequency_combo : float
        Minimum frequency threshold for combination features
    drop_target_encoded_cols : bool
        Whether to drop original categorical columns after target encoding
    original_data_path : str
        Path to original data file
    
    Returns:
    --------
    Tuple of (train_pipeline, test_pipeline)
    """
    
    # Set defaults if not provided
    if categorical_cols_for_ohe is None:
        categorical_cols_for_ohe = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    if feature_combinations is None:
        feature_combinations = [("job", "marital"), ("education", "housing"), ("job", "education")]
    
    # Create the training pipeline (includes data concatenation)
    train_pipeline_steps = []
    
    # Step 1: Concatenate with original data (training only)
    train_pipeline_steps.append(('data_concat', DataConcatenator(original_data_path=original_data_path, is_training=True)))
    
    # Step 2: Target encoding
    train_pipeline_steps.append(('target_encode', TargetEncoder(
        original_data_path=original_data_path, 
        drop_original_cols=drop_target_encoded_cols
    )))
    
    # Step 3: Pdays feature engineering
    train_pipeline_steps.append(('pdays_engineer', PdaysFeatureEngineer(create_bins=True)))
    
    # Create the test pipeline (no data concatenation)
    test_pipeline_steps = []
    
    # Step 1: Just apply day conversion (no concatenation)
    test_pipeline_steps.append(('data_prep', DataConcatenator(original_data_path=original_data_path, is_training=False)))
    
    # Step 2: Target encoding (same as training)
    test_pipeline_steps.append(('target_encode', TargetEncoder(
        original_data_path=original_data_path, 
        drop_original_cols=drop_target_encoded_cols
    )))
    
    # Step 3: Pdays feature engineering (same as training)
    test_pipeline_steps.append(('pdays_engineer', PdaysFeatureEngineer(create_bins=True)))
    
    # Create column transformer for parallel processing of different feature types
    transformers = []
    
    # Identify columns that will exist after previous steps
    passthrough_cols = ['age', 'balance', 'duration', 'campaign', 'previous', 'pdays']
    if not drop_target_encoded_cols:
        passthrough_cols.extend(categorical_cols_for_ohe)
    
    # Add target encoded columns
    target_encoded_cols = [col + "_mean" for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']]
    passthrough_cols.extend(target_encoded_cols)
    
    # Add pdays engineered columns
    pdays_cols = ['pdays_none', 'pdays_qtr_yr', 'pdays_hlf_yr', 'pdays_fl_yr', 'pdays_mor_yr']
    passthrough_cols.extend(pdays_cols)
    
    # Passthrough numerical and target encoded columns
    transformers.append(('passthrough', 'passthrough', passthrough_cols))
    
    # Rare one-hot encoding for categorical columns (if not dropped)
    if not drop_target_encoded_cols and categorical_cols_for_ohe:
        transformers.append(('rare_ohe', RareOneHotEncoder(min_frequency=min_frequency_ohe), categorical_cols_for_ohe))
    
    # Combination features
    if feature_combinations:
        # We need to specify which columns to use for combinations
        # This will use all categorical columns available
        combo_cols = categorical_cols_for_ohe if not drop_target_encoded_cols else []
        if combo_cols:
            transformers.append(('combo_features', CombinationFeatureEncoder(
                feature_combinations=feature_combinations,
                min_frequency=min_frequency_combo
            ), combo_cols))
    
    # Step 4: Column transformer (same for both)
    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not explicitly handled
        sparse_threshold=0  # Return dense arrays
    )
    
    train_pipeline_steps.append(('column_transform', column_transformer))
    test_pipeline_steps.append(('column_transform', column_transformer))
    
    # Create final pipelines
    train_pipeline = Pipeline(train_pipeline_steps)
    test_pipeline = Pipeline(test_pipeline_steps)
    
    return train_pipeline, test_pipeline


def create_single_pipeline(
    categorical_cols_for_ohe=None,
    feature_combinations=None,
    min_frequency_ohe=0.01,
    min_frequency_combo=0.01,
    drop_target_encoded_cols=False,
    original_data_path="bank-full.csv",
    for_training=True
):
    """
    Creates a single pipeline for either training or testing.
    
    Parameters:
    -----------
    for_training : bool
        If True, includes data concatenation. If False, skips it.
    """
    
    # Set defaults if not provided
    if categorical_cols_for_ohe is None:
        categorical_cols_for_ohe = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    if feature_combinations is None:
        feature_combinations = [("job", "marital"), ("education", "housing"), ("job", "education")]
    
    # Create the pipeline
    pipeline_steps = []
    
    # Step 1: Data preparation (concatenate only for training)
    pipeline_steps.append(('data_prep', DataConcatenator(original_data_path=original_data_path, is_training=for_training)))
    
    # Step 2: Target encoding
    pipeline_steps.append(('target_encode', TargetEncoder(
        original_data_path=original_data_path, 
        drop_original_cols=drop_target_encoded_cols
    )))
    
    # Step 3: Pdays feature engineering
    pipeline_steps.append(('pdays_engineer', PdaysFeatureEngineer(create_bins=True)))
    
    # Create column transformer for parallel processing of different feature types
    transformers = []
    
    # Identify columns that will exist after previous steps
    passthrough_cols = ['age', 'balance', 'duration', 'campaign', 'previous', 'pdays']
    if not drop_target_encoded_cols:
        passthrough_cols.extend(categorical_cols_for_ohe)
    
    # Add target encoded columns
    target_encoded_cols = [col + "_mean" for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']]
    passthrough_cols.extend(target_encoded_cols)
    
    # Add pdays engineered columns
    pdays_cols = ['pdays_none', 'pdays_qtr_yr', 'pdays_hlf_yr', 'pdays_fl_yr', 'pdays_mor_yr']
    passthrough_cols.extend(pdays_cols)
    
    # Passthrough numerical and target encoded columns
    transformers.append(('passthrough', 'passthrough', passthrough_cols))
    
    # Rare one-hot encoding for categorical columns (if not dropped)
    if not drop_target_encoded_cols and categorical_cols_for_ohe:
        transformers.append(('rare_ohe', RareOneHotEncoder(min_frequency=min_frequency_ohe), categorical_cols_for_ohe))
    
    # Combination features
    if feature_combinations:
        # We need to specify which columns to use for combinations
        # This will use all categorical columns available
        combo_cols = categorical_cols_for_ohe if not drop_target_encoded_cols else []
        if combo_cols:
            transformers.append(('combo_features', CombinationFeatureEncoder(
                feature_combinations=feature_combinations,
                min_frequency=min_frequency_combo
            ), combo_cols))
    
    # Step 4: Column transformer
    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not explicitly handled
        sparse_threshold=0  # Return dense arrays
    )
    
    pipeline_steps.append(('column_transform', column_transformer))
    
    # Create final pipeline
    pipeline = Pipeline(pipeline_steps)
    
    return pipeline


# Example usage function
def create_example_pipeline():
    """
    Creates an example pipeline with typical settings for your dataset.
    Returns a single pipeline that can be used for both training and test.
    """
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    feature_combinations = [
        ("job", "marital"),
        ("education", "housing"), 
        ("job", "education"),
        ("marital", "housing")
    ]
    
    # Create training pipeline (includes data concatenation)
    train_pipeline = create_single_pipeline(
        categorical_cols_for_ohe=categorical_cols,
        feature_combinations=feature_combinations,
        min_frequency_ohe=0.02,  # Categories appearing in less than 2% of data become 'rare'
        min_frequency_combo=0.01,  # Combinations appearing in less than 1% become 'rare'
        drop_target_encoded_cols=True,  # Drop original categorical columns after target encoding
        original_data_path="bank-full.csv",
        for_training=True
    )
    
    return train_pipeline


class TrainTestAwarePipeline:
    """
    A pipeline wrapper that can handle both training and test data correctly.
    Training data gets concatenated with original data, test data doesn't.
    """
    def __init__(self, 
                 categorical_cols_for_ohe=None,
                 feature_combinations=None,
                 min_frequency_ohe=0.01,
                 min_frequency_combo=0.01,
                 drop_target_encoded_cols=False,
                 original_data_path="bank-full.csv"):
        
        self.categorical_cols_for_ohe = categorical_cols_for_ohe or ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        self.feature_combinations = feature_combinations or [("job", "marital"), ("education", "housing"), ("job", "education")]
        self.min_frequency_ohe = min_frequency_ohe
        self.min_frequency_combo = min_frequency_combo
        self.drop_target_encoded_cols = drop_target_encoded_cols
        self.original_data_path = original_data_path
        
        # Create both pipelines
        self.train_pipeline = None
        self.test_pipeline = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Fit on training data (includes data concatenation)"""
        # Create training pipeline
        self.train_pipeline = create_single_pipeline(
            categorical_cols_for_ohe=self.categorical_cols_for_ohe,
            feature_combinations=self.feature_combinations,
            min_frequency_ohe=self.min_frequency_ohe,
            min_frequency_combo=self.min_frequency_combo,
            drop_target_encoded_cols=self.drop_target_encoded_cols,
            original_data_path=self.original_data_path,
            for_training=True
        )
        
        # Fit training pipeline
        self.train_pipeline.fit(X, y)
        
        # Create test pipeline (no data concatenation) with same fitted transformers
        self.test_pipeline = create_single_pipeline(
            categorical_cols_for_ohe=self.categorical_cols_for_ohe,
            feature_combinations=self.feature_combinations,
            min_frequency_ohe=self.min_frequency_ohe,
            min_frequency_combo=self.min_frequency_combo,
            drop_target_encoded_cols=self.drop_target_encoded_cols,
            original_data_path=self.original_data_path,
            for_training=False
        )
        
        # Fit the test pipeline on a small sample to initialize it, then copy fitted components
        # We need to fit it on something to initialize the structure
        sample_X = X.head(10) if hasattr(X, 'head') else X[:10]
        self.test_pipeline.fit(sample_X, y[:10] if y is not None else None)
        
        # Now copy the fitted transformers from training to test pipeline
        self.test_pipeline.named_steps['target_encode'] = self.train_pipeline.named_steps['target_encode']
        self.test_pipeline.named_steps['column_transform'] = self.train_pipeline.named_steps['column_transform']
        
        self.is_fitted = True
        return self
        
    def transform_train(self, X):
        """Transform training data (includes concatenation)"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming")
        return self.train_pipeline.transform(X)
        
    def transform_test(self, X):
        """Transform test data (no concatenation)"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming")
        return self.test_pipeline.transform(X)
        
    def transform(self, X, is_training=True):
        """Transform data. Use is_training=True for training data, False for test data."""
        if is_training:
            return self.transform_train(X)
        else:
            return self.transform_test(X)


if __name__ == "__main__":
    # Example of how to use the pipeline
    import pandas as pd
    
    # Load your data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # Create and fit training pipeline
    train_pipeline = create_example_pipeline()
    
    # Fit on training data (excluding target and id)
    X_train = train.drop(columns=['y', 'id'])
    y_train = train['y']
    
    train_pipeline.fit(X_train, y_train)
    
    # Transform training data
    X_train_transformed = train_pipeline.transform(X_train)
    
    # Create test pipeline that doesn't concatenate original data
    test_pipeline = create_test_pipeline_from_fitted(train_pipeline)
    
    # Transform test data  
    X_test = test.drop(columns=['id'])
    X_test_transformed = test_pipeline.transform(X_test)
    
    print(f"Original train shape: {X_train.shape}")
    print(f"Transformed train shape: {X_train_transformed.shape}")
    print(f"Original test shape: {X_test.shape}")
    print(f"Transformed test shape: {X_test_transformed.shape}")
    
    print(f"Train columns match test columns: {X_train_transformed.shape[1] == X_test_transformed.shape[1]}")
    print(f"All columns are numerical: {pd.DataFrame(X_train_transformed).select_dtypes(include=[np.number]).shape[1] == X_train_transformed.shape[1]}")
    
    # Show that test data has correct shape (no extra rows from original data)
    print(f"Test data has correct number of rows (no extra original data): {X_test_transformed.shape[0] == test.shape[0]}")
