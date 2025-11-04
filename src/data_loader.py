"""Data loading and preprocessing module."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load the Iris dataset and perform train/test split.

    Args:
        test_size: Proportion of dataset to include in the test split
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test: Split datasets
        feature_names: List of feature names
        target_names: List of target class names
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names


def standardise_features(X_train, X_test=None):
    """
    Standardise features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Test features (optional)

    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features (if provided)
        scaler: Fitted StandardScaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler


def reduce_dimensions(X, n_components=2):
    """
    Apply PCA for dimensionality reduction (for visualisation).

    Args:
        X: Feature matrix
        n_components: Number of principal components

    Returns:
        X_reduced: Reduced feature matrix
        pca: Fitted PCA object
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced, pca
