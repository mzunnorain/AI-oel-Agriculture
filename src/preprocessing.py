"""
Data preprocessing module for agricultural dataset
Handles data loading, cleaning, encoding, and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data preprocessing operations"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.data_stats = {}

    def load_data(self, filepath):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            self.data_stats['original_shape'] = df.shape
            self.data_stats['columns'] = df.columns.tolist()
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def handle_missing_values(self, df, strategy='median'):
        """Handle missing values through imputation"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])

        # For categorical columns, use mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
            df[col] = df[col].fillna(fill_val)

        missing_remaining = df.isnull().sum().sum()
        print(f"Missing values handled. Remaining: {missing_remaining}")
        self.data_stats['missing_values_handled'] = True
        return df

    def detect_and_treat_outliers(self, df, method='iqr', exclude_cols=None):
        """Detect and treat outliers using IQR method.

        Args:
            df: Input DataFrame
            method: Detection method ('iqr')
            exclude_cols: List of columns to skip (e.g. target columns)
        """
        exclude_cols = exclude_cols or []
        numeric_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]
        removed_rows = 0

        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            initial_len = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed_rows += initial_len - len(df)

        print(f"Outliers treated. Rows removed: {removed_rows}")
        self.data_stats['rows_after_outlier_treatment'] = len(df)
        return df

    def encode_categorical_features(self, df, categorical_columns=None):
        """Encode categorical features using LabelEncoder"""
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        print(f"Categorical features encoded: {list(categorical_columns)}")
        return df

    def scale_features(self, df, feature_columns=None, fit=True):
        """Scale numerical features to standard distribution.

        Only scales the specified feature_columns; target columns must be
        excluded by the caller to prevent data leakage.
        """
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if fit:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = self.scaler.transform(df[feature_columns])

        print(f"Features scaled: {len(feature_columns)} numerical features")
        self.feature_names = feature_columns
        return df

    def scale_input(self, X):
        """Scale a single input array using the already-fitted scaler.

        Args:
            X: numpy array of shape (1, n_features)

        Returns:
            Scaled numpy array
        """
        return self.scaler.transform(X)

    def preprocess_pipeline(self, filepath, target_columns=None, categorical_cols=None):
        """Complete preprocessing pipeline.

        Args:
            filepath: Path to CSV dataset
            target_columns: List of target column names to exclude from scaling
                            (e.g. ['label', 'yield'])
            categorical_cols: List of categorical columns to encode

        Returns:
            Preprocessed DataFrame
        """
        print("\n=== Starting Data Preprocessing ===")

        target_columns = target_columns or []

        # Load data
        df = self.load_data(filepath)
        if df is None:
            return None

        # Handle missing values
        df = self.handle_missing_values(df)

        # Treat outliers — exclude target columns
        df = self.detect_and_treat_outliers(df, exclude_cols=target_columns)

        # Encode categorical features
        if categorical_cols:
            df = self.encode_categorical_features(df, categorical_cols)
        else:
            df = self.encode_categorical_features(df)

        # Scale only feature columns (exclude targets to prevent data leakage)
        feature_cols = [col for col in df.columns if col not in target_columns]
        df = self.scale_features(df, feature_cols)

        print("=== Preprocessing Complete ===\n")
        return df

    def get_data_dictionary(self):
        """Return data statistics collected during preprocessing"""
        return self.data_stats

    def save(self, filepath='models/preprocessor.pkl'):
        """Persist the fitted scaler and label encoders to disk."""
        Path('models').mkdir(exist_ok=True)
        payload = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'data_stats': self.data_stats,
        }
        joblib.dump(payload, filepath)
        print(f"Preprocessor saved to {filepath}")

    def load(self, filepath='models/preprocessor.pkl'):
        """Load a previously fitted preprocessor from disk."""
        payload = joblib.load(filepath)
        self.scaler = payload['scaler']
        self.label_encoders = payload['label_encoders']
        self.imputer = payload['imputer']
        self.feature_names = payload.get('feature_names')
        self.data_stats = payload.get('data_stats', {})
        print(f"Preprocessor loaded from {filepath}")
