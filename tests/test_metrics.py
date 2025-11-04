"""Tests for metrics module."""
import pytest
import numpy as np
import pandas as pd
from src.metrics import (
    evaluate_knn,
    evaluate_hdbscan,
    create_comparison_table
)


class TestEvaluateKNN:
    """Tests for evaluate_knn function."""

    def test_evaluate_knn_basic(self):
        """Test basic KNN evaluation."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = evaluate_knn(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert metrics['accuracy'] == 1.0

    def test_evaluate_knn_with_target_names(self):
        """Test KNN evaluation with target names."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        target_names = ['setosa', 'versicolor', 'virginica']

        metrics = evaluate_knn(y_true, y_pred, target_names)

        assert 'classification_report' in metrics
        assert metrics['classification_report'] is not None

    def test_evaluate_knn_accuracy_range(self):
        """Test that accuracy is between 0 and 1."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([1, 0, 2, 2, 1, 0])

        metrics = evaluate_knn(y_true, y_pred)

        assert 0 <= metrics['accuracy'] <= 1

    def test_evaluate_knn_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = evaluate_knn(y_true, y_pred)

        conf_matrix = metrics['confusion_matrix']
        assert conf_matrix.shape == (3, 3)

    def test_evaluate_knn_imperfect_predictions(self):
        """Test KNN evaluation with some incorrect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 2, 2])  # 2 errors

        metrics = evaluate_knn(y_true, y_pred)

        assert metrics['accuracy'] < 1.0
        assert metrics['accuracy'] > 0.5

    def test_evaluate_knn_classification_report_structure(self):
        """Test classification report has expected structure."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        target_names = ['setosa', 'versicolor', 'virginica']

        metrics = evaluate_knn(y_true, y_pred, target_names)

        report = metrics['classification_report']
        assert 'setosa' in report
        assert 'accuracy' in report


class TestEvaluateHDBSCAN:
    """Tests for evaluate_hdbscan function."""

    def test_evaluate_hdbscan_basic(self):
        """Test basic HDBSCAN evaluation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        metrics = evaluate_hdbscan(y_true, y_pred)

        assert 'ari' in metrics
        assert 'nmi' in metrics
        assert 'n_clusters' in metrics
        assert 'n_noise' in metrics

    def test_evaluate_hdbscan_perfect_clustering(self):
        """Test HDBSCAN evaluation with perfect clustering."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        metrics = evaluate_hdbscan(y_true, y_pred)

        # Perfect clustering should have ARI and NMI of 1.0
        assert metrics['ari'] == 1.0
        assert metrics['nmi'] == 1.0
        assert metrics['n_clusters'] == 3
        assert metrics['n_noise'] == 0

    def test_evaluate_hdbscan_with_noise(self):
        """Test HDBSCAN evaluation with noise points."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 0])
        y_pred = np.array([0, 0, 1, 1, 2, 2, -1])  # Last point is noise

        metrics = evaluate_hdbscan(y_true, y_pred)

        assert metrics['n_noise'] == 1
        assert metrics['n_clusters'] == 3

    def test_evaluate_hdbscan_metric_ranges(self):
        """Test that ARI and NMI are in valid ranges."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 0, 0, 2, 2])

        metrics = evaluate_hdbscan(y_true, y_pred)

        # ARI can be negative, but typically between -1 and 1
        assert -1 <= metrics['ari'] <= 1
        # NMI should be between 0 and 1
        assert 0 <= metrics['nmi'] <= 1

    def test_evaluate_hdbscan_random_clustering(self):
        """Test HDBSCAN with random clustering."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 1, 0, 2, 1, 0])

        metrics = evaluate_hdbscan(y_true, y_pred)

        # Random clustering should have low scores
        assert metrics['ari'] < 0.5
        assert metrics['nmi'] < 0.7

    def test_evaluate_hdbscan_single_cluster(self):
        """Test HDBSCAN when all points in one cluster."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        metrics = evaluate_hdbscan(y_true, y_pred)

        assert metrics['n_clusters'] == 1
        assert metrics['n_noise'] == 0

    def test_evaluate_hdbscan_all_noise(self):
        """Test HDBSCAN when all points are noise."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([-1, -1, -1, -1, -1, -1])

        metrics = evaluate_hdbscan(y_true, y_pred)

        assert metrics['n_clusters'] == 0
        assert metrics['n_noise'] == 6


class TestCreateComparisonTable:
    """Tests for create_comparison_table function."""

    def test_create_comparison_table_basic(self):
        """Test basic comparison table creation."""
        knn_metrics = {
            'accuracy': 0.95,
            'confusion_matrix': np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        }
        hdbscan_metrics = {
            'ari': 0.85,
            'nmi': 0.90,
            'n_clusters': 3,
            'n_noise': 2
        }

        df = create_comparison_table(knn_metrics, hdbscan_metrics)

        assert isinstance(df, pd.DataFrame)
        assert 'Metric' in df.columns
        assert 'Value' in df.columns

    def test_create_comparison_table_has_all_metrics(self):
        """Test that comparison table includes all expected metrics."""
        knn_metrics = {
            'accuracy': 0.95,
            'confusion_matrix': np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        }
        hdbscan_metrics = {
            'ari': 0.85,
            'nmi': 0.90,
            'n_clusters': 3,
            'n_noise': 2
        }

        df = create_comparison_table(knn_metrics, hdbscan_metrics)

        # Check that expected metrics are in the table
        metrics_list = df['Metric'].tolist()
        assert any('Accuracy' in m for m in metrics_list)
        assert any('ARI' in m for m in metrics_list)
        assert any('NMI' in m for m in metrics_list)
        assert any('Clusters' in m for m in metrics_list)
        assert any('Noise' in m for m in metrics_list)

    def test_create_comparison_table_row_count(self):
        """Test that comparison table has correct number of rows."""
        knn_metrics = {
            'accuracy': 0.95,
            'confusion_matrix': np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        }
        hdbscan_metrics = {
            'ari': 0.85,
            'nmi': 0.90,
            'n_clusters': 3,
            'n_noise': 2
        }

        df = create_comparison_table(knn_metrics, hdbscan_metrics)

        # Should have 5 rows (1 KNN metric, 4 HDBSCAN metrics)
        assert len(df) == 5

    def test_create_comparison_table_values_formatted(self):
        """Test that values are properly formatted."""
        knn_metrics = {
            'accuracy': 0.954321,
            'confusion_matrix': np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        }
        hdbscan_metrics = {
            'ari': 0.856789,
            'nmi': 0.901234,
            'n_clusters': 3,
            'n_noise': 2
        }

        df = create_comparison_table(knn_metrics, hdbscan_metrics)

        # Check that float values are formatted to 4 decimal places
        values = df['Value'].tolist()
        assert any('0.9543' in str(v) for v in values)
        assert any('0.8568' in str(v) for v in values)

    def test_create_comparison_table_integer_metrics(self):
        """Test that integer metrics are preserved."""
        knn_metrics = {
            'accuracy': 0.95,
            'confusion_matrix': np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        }
        hdbscan_metrics = {
            'ari': 0.85,
            'nmi': 0.90,
            'n_clusters': 3,
            'n_noise': 5
        }

        df = create_comparison_table(knn_metrics, hdbscan_metrics)

        values = df['Value'].tolist()
        # Check that integers are present
        assert 3 in values or '3' in values
        assert 5 in values or '5' in values
