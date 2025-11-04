"""Model training and prediction module."""
from sklearn.neighbors import KNeighborsClassifier
import hdbscan
import numpy as np


class KNNModel:
    """K-Nearest Neighbors classifier wrapper."""

    def __init__(self, n_neighbours=5, metric='euclidean'):
        """
        Initialise KNN model.

        Args:
            n_neighbours: Number of neighbours to use
            metric: Distance metric to use
        """
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbours,
            metric=metric
        )

    def train(self, X_train, y_train):
        """Train the KNN model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict labels for test data."""
        return self.model.predict(X_test)

    def get_accuracy(self, X_test, y_test):
        """Calculate accuracy on test data."""
        return self.model.score(X_test, y_test)


class HDBSCANModel:
    """HDBSCAN clustering wrapper."""

    def __init__(self, min_cluster_size=5, min_samples=None):
        """
        Initialise HDBSCAN model.

        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum number of samples in a neighbourhood
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            gen_min_span_tree=True
        )

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.

        Returns:
            labels: Cluster labels (-1 for noise points)
        """
        labels = self.model.fit_predict(X)
        return labels

    def get_n_clusters(self):
        """Get the number of clusters found (excluding noise)."""
        if hasattr(self.model, 'labels_'):
            return len(set(self.model.labels_)) - (1 if -1 in self.model.labels_ else 0)
        return 0
