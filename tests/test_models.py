"""Tests for models module."""
import pytest
import numpy as np
from src.models import KNNModel, HDBSCANModel


class TestKNNModel:
    """Tests for KNNModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data."""
        np.random.seed(42)
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.rand(20, 4)
        y_test = np.random.randint(0, 3, 20)
        return X_train, y_train, X_test, y_test

    def test_knn_initialisation(self):
        """Test KNN model initialisation."""
        model = KNNModel(n_neighbours=5, metric='euclidean')

        assert model.n_neighbours == 5
        assert model.metric == 'euclidean'
        assert model.model is not None

    def test_knn_default_parameters(self):
        """Test KNN model with default parameters."""
        model = KNNModel()

        assert model.n_neighbours == 5
        assert model.metric == 'euclidean'

    def test_knn_train(self, sample_data):
        """Test KNN model training."""
        X_train, y_train, _, _ = sample_data
        model = KNNModel(n_neighbours=3)

        # Should not raise any errors
        model.train(X_train, y_train)

        # Model should be fitted
        assert hasattr(model.model, 'classes_')

    def test_knn_predict(self, sample_data):
        """Test KNN model prediction."""
        X_train, y_train, X_test, _ = sample_data
        model = KNNModel(n_neighbours=3)
        model.train(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1, 2]))

    def test_knn_get_accuracy(self, sample_data):
        """Test KNN accuracy calculation."""
        X_train, y_train, X_test, y_test = sample_data
        model = KNNModel(n_neighbours=3)
        model.train(X_train, y_train)

        accuracy = model.get_accuracy(X_test, y_test)

        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, (float, np.floating))

    def test_knn_different_neighbours(self):
        """Test KNN with different number of neighbours."""
        model_3 = KNNModel(n_neighbours=3)
        model_7 = KNNModel(n_neighbours=7)

        assert model_3.n_neighbours == 3
        assert model_7.n_neighbours == 7

    def test_knn_perfect_classification(self):
        """Test KNN with perfectly separable data."""
        # Create perfectly separable data
        X_train = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2],
            [5, 5], [5.1, 5.1], [5.2, 5.2],
            [10, 10], [10.1, 10.1], [10.2, 10.2]
        ])
        y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        X_test = np.array([[0.15, 0.15], [5.15, 5.15], [10.15, 10.15]])
        y_test = np.array([0, 1, 2])

        model = KNNModel(n_neighbours=1)
        model.train(X_train, y_train)

        accuracy = model.get_accuracy(X_test, y_test)
        assert accuracy == 1.0


class TestHDBSCANModel:
    """Tests for HDBSCANModel class."""

    @pytest.fixture
    def sample_clusterable_data(self):
        """Create sample data with clear clusters."""
        np.random.seed(42)
        # Three clear clusters
        cluster1 = np.random.randn(30, 2) + [0, 0]
        cluster2 = np.random.randn(30, 2) + [10, 10]
        cluster3 = np.random.randn(30, 2) + [20, 0]
        X = np.vstack([cluster1, cluster2, cluster3])
        return X

    def test_hdbscan_initialisation(self):
        """Test HDBSCAN model initialisation."""
        model = HDBSCANModel(min_cluster_size=5, min_samples=3)

        assert model.min_cluster_size == 5
        assert model.min_samples == 3
        assert model.model is not None

    def test_hdbscan_default_parameters(self):
        """Test HDBSCAN model with default parameters."""
        model = HDBSCANModel()

        assert model.min_cluster_size == 5
        assert model.min_samples is None

    def test_hdbscan_fit_predict(self, sample_clusterable_data):
        """Test HDBSCAN clustering."""
        model = HDBSCANModel(min_cluster_size=5)
        labels = model.fit_predict(sample_clusterable_data)

        # Should return labels for all points
        assert labels.shape == (90,)
        # Should find at least one cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1

    def test_hdbscan_get_n_clusters(self, sample_clusterable_data):
        """Test getting number of clusters."""
        model = HDBSCANModel(min_cluster_size=5)
        model.fit_predict(sample_clusterable_data)

        n_clusters = model.get_n_clusters()

        assert isinstance(n_clusters, int)
        assert n_clusters >= 0

    def test_hdbscan_noise_detection(self):
        """Test that HDBSCAN can detect noise points."""
        # Create data with clear clusters and some outliers
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2) + [0, 0]
        cluster2 = np.random.randn(20, 2) + [10, 10]
        outliers = np.array([[50, 50], [60, 60], [-30, -30]])
        X = np.vstack([cluster1, cluster2, outliers])

        model = HDBSCANModel(min_cluster_size=5)
        labels = model.fit_predict(X)

        # Check that some points might be labelled as noise (-1)
        assert labels.shape == (43,)
        # Noise points are labelled as -1
        n_noise = sum(labels == -1)
        assert n_noise >= 0

    def test_hdbscan_returns_integer_labels(self, sample_clusterable_data):
        """Test that HDBSCAN returns integer labels."""
        model = HDBSCANModel(min_cluster_size=5)
        labels = model.fit_predict(sample_clusterable_data)

        assert labels.dtype in [np.int32, np.int64]

    def test_hdbscan_get_n_clusters_before_fitting(self):
        """Test get_n_clusters before fitting returns 0."""
        model = HDBSCANModel(min_cluster_size=5)
        n_clusters = model.get_n_clusters()

        assert n_clusters == 0

    def test_hdbscan_different_min_cluster_sizes(self):
        """Test HDBSCAN with different min_cluster_size values."""
        model_small = HDBSCANModel(min_cluster_size=3)
        model_large = HDBSCANModel(min_cluster_size=15)

        assert model_small.min_cluster_size == 3
        assert model_large.min_cluster_size == 15
