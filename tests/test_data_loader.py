"""Tests for data_loader module."""
import pytest
import numpy as np
from src.data_loader import (
    load_and_preprocess_data,
    standardise_features,
    reduce_dimensions
)


class TestLoadAndPreprocessData:
    """Tests for load_and_preprocess_data function."""

    def test_load_returns_correct_shapes(self):
        """Test that data loading returns correct shapes."""
        X_train, X_test, y_train, y_test, feature_names, target_names = load_and_preprocess_data(
            test_size=0.2, random_state=42
        )

        # Check shapes
        assert X_train.shape[0] == 120  # 80% of 150
        assert X_test.shape[0] == 30    # 20% of 150
        assert X_train.shape[1] == 4    # 4 features
        assert X_test.shape[1] == 4
        assert y_train.shape[0] == 120
        assert y_test.shape[0] == 30

    def test_load_returns_correct_feature_names(self):
        """Test that correct feature names are returned."""
        _, _, _, _, feature_names, _ = load_and_preprocess_data()

        assert len(feature_names) == 4
        assert 'sepal' in feature_names[0].lower()
        assert 'petal' in feature_names[2].lower()

    def test_load_returns_correct_target_names(self):
        """Test that correct target names are returned."""
        _, _, _, _, _, target_names = load_and_preprocess_data()

        assert len(target_names) == 3
        assert 'setosa' in target_names
        assert 'versicolor' in target_names
        assert 'virginica' in target_names

    def test_stratified_split(self):
        """Test that split maintains class proportions."""
        X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data(
            test_size=0.2, random_state=42
        )

        # Check each class is present in both splits
        train_classes = set(y_train)
        test_classes = set(y_test)

        assert train_classes == test_classes == {0, 1, 2}

    def test_random_state_reproducibility(self):
        """Test that same random state produces same split."""
        X_train1, X_test1, _, _, _, _ = load_and_preprocess_data(random_state=42)
        X_train2, X_test2, _, _, _, _ = load_and_preprocess_data(random_state=42)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)


class TestStandardiseFeatures:
    """Tests for standardise_features function."""

    def test_standardise_transforms_data(self):
        """Test that standardisation transforms data correctly."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled, scaler = standardise_features(X)

        # Mean should be close to 0
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        # Standard deviation should be close to 1
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)

    def test_standardise_with_train_test(self):
        """Test standardisation with separate train/test sets."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[2, 3], [4, 5]])

        X_train_scaled, X_test_scaled, scaler = standardise_features(X_train, X_test)

        # Check shapes are preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

        # Train data should have mean ~0 and std ~1
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)

    def test_scaler_object_returned(self):
        """Test that scaler object is returned."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        _, scaler = standardise_features(X)

        assert scaler is not None
        assert hasattr(scaler, 'transform')


class TestReduceDimensions:
    """Tests for reduce_dimensions function."""

    def test_reduce_dimensions_correct_shape(self):
        """Test that PCA reduces to correct number of components."""
        X = np.random.rand(100, 4)
        X_reduced, pca = reduce_dimensions(X, n_components=2)

        assert X_reduced.shape == (100, 2)

    def test_pca_object_returned(self):
        """Test that PCA object is returned."""
        X = np.random.rand(100, 4)
        _, pca = reduce_dimensions(X, n_components=2)

        assert pca is not None
        assert hasattr(pca, 'transform')
        assert hasattr(pca, 'explained_variance_ratio_')

    def test_reduce_dimensions_preserves_rows(self):
        """Test that number of samples is preserved."""
        X = np.random.rand(50, 4)
        X_reduced, _ = reduce_dimensions(X, n_components=3)

        assert X_reduced.shape[0] == X.shape[0]
        assert X_reduced.shape[1] == 3

    def test_explained_variance_exists(self):
        """Test that explained variance is computed."""
        X = np.random.rand(100, 4)
        _, pca = reduce_dimensions(X, n_components=2)

        assert len(pca.explained_variance_ratio_) == 2
        assert np.all(pca.explained_variance_ratio_ >= 0)
        assert np.all(pca.explained_variance_ratio_ <= 1)
