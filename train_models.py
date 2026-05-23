"""
Training Pipeline for Smart Agriculture Decision Support System
Trains all three ML models and saves serialized artifacts to models/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import DataPreprocessor
from models import ModelFactory
from utils import save_metrics, create_directories


# ---------------------------------------------------------------------------
# Configuration — update these if your dataset uses different column names
# ---------------------------------------------------------------------------
DATA_PATH = 'data/agricultural_data.csv'
TARGET_CLASSIFIER = 'label'   # crop type column
TARGET_REGRESSOR  = 'yield'   # yield column


# ---------------------------------------------------------------------------
def load_and_preprocess(data_path: str, preprocessor: DataPreprocessor):
    """Load CSV and run the full preprocessing pipeline.

    Target columns are excluded from scaling to prevent data leakage.
    """
    print("Loading and preprocessing data…")
    df_raw = pd.read_csv(data_path)
    print(f"Original dataset shape: {df_raw.shape}")

    # Identify categorical columns (excluding targets)
    categorical_cols = [
        c for c in df_raw.select_dtypes(include='object').columns
        if c not in (TARGET_CLASSIFIER, TARGET_REGRESSOR)
    ]

    df = preprocessor.preprocess_pipeline(
        data_path,
        target_columns=[TARGET_CLASSIFIER, TARGET_REGRESSOR],
        categorical_cols=categorical_cols if categorical_cols else None,
    )
    return df


# ---------------------------------------------------------------------------
def train_all_models(df: pd.DataFrame, preprocessor: DataPreprocessor):
    """Train Decision Tree, KMeans, and Linear Regression models."""
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    factory = ModelFactory()

    # Feature matrix (scaled) — exclude both targets
    feature_cols = [c for c in df.columns
                    if c not in (TARGET_CLASSIFIER, TARGET_REGRESSOR)]
    X = df[feature_cols]

    # Targets (not scaled)
    y_class = df[TARGET_CLASSIFIER]
    y_reg   = df[TARGET_REGRESSOR]

    # For regression, include the encoded crop label as a feature
    # (yield is strongly crop-type dependent)
    X_reg = df[[c for c in df.columns if c != TARGET_REGRESSOR]]

    # Train / test split — stratify on crop label for balanced evaluation
    X_train, X_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # 1. Decision Tree Classifier
    print("\n1. Training Decision Tree Classifier…")
    factory.dt_classifier.train(X_train, yc_train)
    dt_metrics = factory.dt_classifier.evaluate(X_test, yc_test)

    # 2. KMeans Clustering (unsupervised — use full training features)
    print("\n2. Training KNN Clustering (K-Means)…")
    factory.kmeans_clusterer.train(X_train)
    kmeans_metrics = factory.kmeans_clusterer.evaluate(X_train)

    # 3. Linear Regression (uses crop label as a feature — yield is crop-type dependent)
    print("\n3. Training Linear Regression…")
    factory.lr_regressor.train(Xr_train, yr_train)
    lr_metrics = factory.lr_regressor.evaluate(Xr_test, yr_test)

    # Persist models
    factory.save_all_models()
    preprocessor.save()   # save fitted scaler + label encoders for GUI use

    all_metrics = {
        'decision_tree':    dt_metrics,
        'kmeans':           kmeans_metrics,
        'linear_regression': lr_metrics,
    }
    save_metrics(all_metrics, 'results/model_metrics.json')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return factory, all_metrics


# ---------------------------------------------------------------------------
def print_report(all_metrics: dict):
    """Print a formatted summary of all model metrics."""
    print("\n\n=== MODEL TRAINING REPORT ===")

    print("\nDecision Tree Classifier:")
    for k, v in all_metrics['decision_tree'].items():
        print(f"  {k:<20}: {v:.4f}")

    print("\nKNN Clustering (K-Means):")
    for k, v in all_metrics['kmeans'].items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    print("\nLinear Regression:")
    for k, v in all_metrics['linear_regression'].items():
        print(f"  {k:<20}: {v:.4f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    create_directories()

    preprocessor = DataPreprocessor()

    try:
        df = load_and_preprocess(DATA_PATH, preprocessor)
        if df is None or df.empty:
            raise RuntimeError("Preprocessing returned empty DataFrame.")

        factory, metrics = train_all_models(df, preprocessor)
        print_report(metrics)

        print("\n✓ All models trained and saved successfully!")
        print("✓ Launch the GUI:  python src/gui.py")

    except FileNotFoundError:
        print(f"\n✗ Dataset not found at '{DATA_PATH}'.")
        print("  Run 'python prepare_dataset.py' first to generate a sample dataset.")
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n✗ Error during training: {exc}")
        traceback.print_exc()
        sys.exit(1)
