"""
Machine Learning Models Module
Implements Decision Tree Classification, KNN Clustering (K-Means), and Linear Regression
as required by the OEL specification.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, mean_squared_error, mean_absolute_error, r2_score,
    davies_bouldin_score
)
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Cluster agronomic guidance lookup
# ---------------------------------------------------------------------------
_CLUSTER_GUIDANCE = {
    0: {
        'label': 'High Fertility Zone',
        'guidance': (
            'High NPK levels detected. Suitable for high-value crops '
            '(vegetables, fruits, spices). Maintain organic matter and monitor pH.'
        ),
    },
    1: {
        'label': 'Medium Fertility Zone',
        'guidance': (
            'Moderate nutrient levels. Optimal for staple grains (rice, wheat) '
            'and pulses. Apply seasonal fertilization and practice crop rotation.'
        ),
    },
    2: {
        'label': 'Low Fertility Zone',
        'guidance': (
            'Low NPK levels. Prioritize soil amendment. Nitrogen-fixing legumes '
            '(chickpea, lentil) recommended. Use precision fertilization.'
        ),
    },
}


def get_cluster_guidance(cluster_id: int) -> dict:
    """Return agronomic guidance for a given cluster ID.

    Falls back to a generic message for unexpected cluster IDs.
    """
    return _CLUSTER_GUIDANCE.get(
        cluster_id,
        {
            'label': f'Zone {cluster_id}',
            'guidance': 'Optimize management practices based on local soil tests.',
        },
    )


# ---------------------------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------------------------
class DecisionTreeModel:
    """Decision Tree Classifier for crop recommendation."""

    def __init__(self, max_depth=10, random_state=42,
                 min_samples_split=5, min_samples_leaf=2):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            criterion='gini',
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.metrics = {}
        self.feature_importance = None
        self.feature_names = None
        self.classes_ = None

    def train(self, X_train, y_train):
        """Train Decision Tree classifier."""
        self.model.fit(X_train, y_train)
        self.feature_names = (
            list(X_train.columns)
            if hasattr(X_train, 'columns')
            else [f'Feature_{i}' for i in range(X_train.shape[1])]
        )
        self.feature_importance = self.model.feature_importances_
        self.classes_ = self.model.classes_
        print("Decision Tree Classifier trained successfully")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance and return metrics dict."""
        y_pred = self.model.predict(X_test)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        print("\n=== Decision Tree Classifier Metrics ===")
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("=" * 40)

        return self.metrics

    def predict(self, X):
        """Return predicted class labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """Return {feature_name: importance} dict."""
        if self.feature_importance is not None:
            return dict(zip(self.feature_names, self.feature_importance))
        return {}

    def save(self, filepath='models/decision_tree_model.pkl'):
        """Serialize trained model to disk."""
        Path('models').mkdir(exist_ok=True)
        payload = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'classes_': self.classes_,
        }
        joblib.dump(payload, filepath)
        print(f"Decision Tree model saved to {filepath}")

    def load(self, filepath='models/decision_tree_model.pkl'):
        """Load serialized model from disk."""
        payload = joblib.load(filepath)
        self.model = payload['model']
        self.feature_names = payload['feature_names']
        self.feature_importance = payload['feature_importance']
        self.classes_ = payload.get('classes_')
        print(f"Decision Tree model loaded from {filepath}")


# ---------------------------------------------------------------------------
# KNN Clustering Model (K-Means)
# ---------------------------------------------------------------------------
class KNNClusteringModel:
    """K-Means Clustering for soil profile segmentation.

    Named 'KNN Clustering' per the OEL specification which uses the term
    'K-Nearest Neighbors Clustering' to refer to K-Means-based zone segmentation.
    """

    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state,
            algorithm='lloyd',
        )
        self.metrics = {}
        self.cluster_centers = None
        self.n_clusters = n_clusters
        self.labels_ = None
        self.X_train_ = None  # stored for visualization

    def train(self, X):
        """Fit K-Means clustering model."""
        self.model.fit(X)
        self.cluster_centers = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        # Store a numpy copy for later scatter-plot use
        self.X_train_ = X.values if hasattr(X, 'values') else np.array(X)
        print(f"KMeans Clustering trained with {self.n_clusters} clusters")

    def evaluate(self, X):
        """Compute and return clustering quality metrics."""
        labels = self.model.labels_
        X_arr = X.values if hasattr(X, 'values') else np.array(X)

        silhouette_avg = silhouette_score(X_arr, labels)
        db_index = davies_bouldin_score(X_arr, labels)

        self.metrics = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin_index': db_index,
            'n_clusters': self.n_clusters,
            'inertia': float(self.model.inertia_),
        }

        print("\n=== KMeans Clustering Metrics ===")
        print(f"  Silhouette Score:      {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index:  {db_index:.4f}")
        print(f"  Inertia:               {self.model.inertia_:.4f}")
        print("=" * 40)

        return self.metrics

    def predict(self, X):
        """Assign cluster labels to new samples."""
        return self.model.predict(X)

    def get_cluster_info(self):
        """Return cluster centers and metadata."""
        return {
            'cluster_centers': self.cluster_centers,
            'n_clusters': self.n_clusters,
            'inertia': self.model.inertia_,
        }

    def save(self, filepath='models/kmeans_model.pkl'):
        """Serialize trained model to disk."""
        Path('models').mkdir(exist_ok=True)
        payload = {
            'model': self.model,
            'cluster_centers': self.cluster_centers,
            'n_clusters': self.n_clusters,
            'labels_': self.labels_,
            'X_train_': self.X_train_,
        }
        joblib.dump(payload, filepath)
        print(f"KMeans model saved to {filepath}")

    def load(self, filepath='models/kmeans_model.pkl'):
        """Load serialized model from disk."""
        payload = joblib.load(filepath)
        self.model = payload['model']
        self.cluster_centers = payload['cluster_centers']
        self.n_clusters = payload['n_clusters']
        self.labels_ = payload.get('labels_')
        self.X_train_ = payload.get('X_train_')
        print(f"KMeans model loaded from {filepath}")


# ---------------------------------------------------------------------------
# Linear Regression Model
# ---------------------------------------------------------------------------
class LinearRegressionModel:
    """Linear Regression for crop yield prediction."""

    def __init__(self):
        self.model = LinearRegression(fit_intercept=True)
        self.metrics = {}
        self.residuals = None
        self.predictions = None
        self.y_test_ = None
        self.feature_names = None
        self._residual_std = None  # used for confidence interval

    def train(self, X_train, y_train):
        """Train Linear Regression model."""
        self.model.fit(X_train, y_train)
        self.feature_names = (
            list(X_train.columns)
            if hasattr(X_train, 'columns')
            else [f'Feature_{i}' for i in range(X_train.shape[1])]
        )
        print("Linear Regression model trained successfully")

    def evaluate(self, X_test, y_test):
        """Evaluate model and return metrics dict."""
        y_pred = self.model.predict(X_test)
        self.predictions = y_pred
        y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        self.residuals = y_test_arr - y_pred
        self.y_test_ = y_test_arr
        self._residual_std = float(np.std(self.residuals))

        self.metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test_arr, y_pred))),
            'mae': float(mean_absolute_error(y_test_arr, y_pred)),
            'r2_score': float(r2_score(y_test_arr, y_pred)),
        }

        print("\n=== Linear Regression Metrics ===")
        print(f"  RMSE:     {self.metrics['rmse']:.4f}")
        print(f"  MAE:      {self.metrics['mae']:.4f}")
        print(f"  R² Score: {self.metrics['r2_score']:.4f}")
        print("=" * 40)

        return self.metrics

    def predict(self, X):
        """Return predicted yield values."""
        return self.model.predict(X)

    def predict_with_ci(self, X, z=1.96):
        """Return (point_estimate, lower_bound, upper_bound) using residual std.

        Uses ±z * residual_std as a simple approximate 95% confidence interval.
        Falls back to ±15% if the model has not been evaluated yet.
        """
        y_pred = self.model.predict(X)[0]
        if self._residual_std is not None and self._residual_std > 0:
            margin = z * self._residual_std
        else:
            margin = abs(y_pred) * 0.15
        return y_pred, y_pred - margin, y_pred + margin

    def get_residuals(self):
        """Return residual array."""
        return self.residuals

    def get_coefficients(self):
        """Return {feature_name: coefficient} dict."""
        if self.model.coef_ is not None and self.feature_names:
            return dict(zip(self.feature_names, self.model.coef_))
        return {}

    def save(self, filepath='models/linear_regression_model.pkl'):
        """Serialize trained model to disk."""
        Path('models').mkdir(exist_ok=True)
        payload = {
            'model': self.model,
            'feature_names': self.feature_names,
            'residuals': self.residuals,
            'predictions': self.predictions,
            'y_test_': self.y_test_,
            '_residual_std': self._residual_std,
        }
        joblib.dump(payload, filepath)
        print(f"Linear Regression model saved to {filepath}")

    def load(self, filepath='models/linear_regression_model.pkl'):
        """Load serialized model from disk."""
        payload = joblib.load(filepath)
        self.model = payload['model']
        self.feature_names = payload['feature_names']
        self.residuals = payload.get('residuals')
        self.predictions = payload.get('predictions')
        self.y_test_ = payload.get('y_test_')
        self._residual_std = payload.get('_residual_std')
        print(f"Linear Regression model loaded from {filepath}")


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
class ModelFactory:
    """Factory class for creating and managing all three models."""

    def __init__(self):
        self.dt_classifier = DecisionTreeModel()
        self.kmeans_clusterer = KNNClusteringModel(n_clusters=3)
        self.lr_regressor = LinearRegressionModel()

    def save_all_models(self):
        """Serialize all trained models to the models/ directory."""
        self.dt_classifier.save()
        self.kmeans_clusterer.save()
        self.lr_regressor.save()
        print("\nAll models saved successfully")

    def load_all_models(self):
        """Load all serialized models. Returns True on success."""
        try:
            self.dt_classifier.load()
            self.kmeans_clusterer.load()
            self.lr_regressor.load()
            print("All models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def get_all_metrics(self):
        """Return metrics dict for all three models."""
        return {
            'decision_tree': self.dt_classifier.metrics,
            'kmeans': self.kmeans_clusterer.metrics,
            'linear_regression': self.lr_regressor.metrics,
        }
