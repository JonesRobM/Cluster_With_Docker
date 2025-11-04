"""Shared pytest fixtures for all tests."""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for all tests


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_iris_data():
    """Load Iris dataset for testing."""
    from src.data_loader import load_and_preprocess_data
    return load_and_preprocess_data(test_size=0.2, random_state=42)


@pytest.fixture
def sample_standardised_data(sample_iris_data):
    """Load and standardise Iris dataset."""
    from src.data_loader import standardise_features
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_iris_data
    X_train_scaled, X_test_scaled, scaler = standardise_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically clean up matplotlib figures after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')
