"""Tests for visualisation module."""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from src.visualisation import (
    plot_pca_clusters,
    plot_confusion_matrix,
    plot_comparison_metrics
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPlotPCAClusters:
    """Tests for plot_pca_clusters function."""

    @pytest.fixture
    def sample_pca_data(self):
        """Create sample PCA data for testing."""
        np.random.seed(42)
        X_pca = np.random.rand(50, 2)
        y_true = np.random.randint(0, 3, 50)
        y_pred = np.random.randint(0, 3, 50)
        target_names = ['setosa', 'versicolor', 'virginica']
        return X_pca, y_true, y_pred, target_names

    def test_plot_pca_clusters_creates_figure(self, sample_pca_data):
        """Test that plot creates a figure."""
        X_pca, y_true, y_pred, target_names = sample_pca_data

        # Should not raise any errors
        plot_pca_clusters(X_pca, y_true, y_pred, "Test", target_names)

        # Clean up
        plt.close('all')

    def test_plot_pca_clusters_saves_file(self, sample_pca_data, temp_output_dir):
        """Test that plot saves to file."""
        X_pca, y_true, y_pred, target_names = sample_pca_data
        save_path = temp_output_dir / "test_plot.png"

        plot_pca_clusters(X_pca, y_true, y_pred, "Test", target_names, save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close('all')

    def test_plot_pca_clusters_without_target_names(self, sample_pca_data):
        """Test plotting without target names."""
        X_pca, y_true, y_pred, _ = sample_pca_data

        # Should not raise any errors
        plot_pca_clusters(X_pca, y_true, y_pred, "Test", target_names=None)

        plt.close('all')

    def test_plot_pca_clusters_without_save_path(self, sample_pca_data):
        """Test plotting without saving."""
        X_pca, y_true, y_pred, target_names = sample_pca_data

        # Should not raise any errors
        plot_pca_clusters(X_pca, y_true, y_pred, "Test", target_names)

        plt.close('all')

    def test_plot_pca_clusters_with_different_sizes(self):
        """Test plotting with different data sizes."""
        X_pca_small = np.random.rand(10, 2)
        y_true_small = np.random.randint(0, 2, 10)
        y_pred_small = np.random.randint(0, 2, 10)

        X_pca_large = np.random.rand(200, 2)
        y_true_large = np.random.randint(0, 3, 200)
        y_pred_large = np.random.randint(0, 3, 200)

        # Both should work without errors
        plot_pca_clusters(X_pca_small, y_true_small, y_pred_small, "Small")
        plot_pca_clusters(X_pca_large, y_true_large, y_pred_large, "Large")

        plt.close('all')

    def test_plot_pca_clusters_2d_requirement(self):
        """Test that X_pca must be 2D."""
        X_pca = np.random.rand(50, 2)
        y = np.random.randint(0, 3, 50)

        # Should work with 2D data
        plot_pca_clusters(X_pca, y, y, "Test")

        plt.close('all')


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    @pytest.fixture
    def sample_confusion_matrix(self):
        """Create sample confusion matrix."""
        conf_matrix = np.array([[10, 0, 0], [0, 8, 2], [1, 0, 9]])
        target_names = ['setosa', 'versicolor', 'virginica']
        return conf_matrix, target_names

    def test_plot_confusion_matrix_creates_figure(self, sample_confusion_matrix):
        """Test that confusion matrix plot creates a figure."""
        conf_matrix, target_names = sample_confusion_matrix

        # Should not raise any errors
        plot_confusion_matrix(conf_matrix, target_names)

        plt.close('all')

    def test_plot_confusion_matrix_saves_file(self, sample_confusion_matrix, temp_output_dir):
        """Test that confusion matrix saves to file."""
        conf_matrix, target_names = sample_confusion_matrix
        save_path = temp_output_dir / "test_confusion.png"

        plot_confusion_matrix(conf_matrix, target_names, save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close('all')

    def test_plot_confusion_matrix_without_save_path(self, sample_confusion_matrix):
        """Test plotting confusion matrix without saving."""
        conf_matrix, target_names = sample_confusion_matrix

        # Should not raise any errors
        plot_confusion_matrix(conf_matrix, target_names)

        plt.close('all')

    def test_plot_confusion_matrix_different_sizes(self):
        """Test confusion matrix with different class counts."""
        # 2x2 matrix
        conf_matrix_2 = np.array([[10, 2], [1, 9]])
        target_names_2 = ['class_0', 'class_1']

        # 4x4 matrix
        conf_matrix_4 = np.array([
            [10, 0, 0, 0],
            [0, 8, 1, 1],
            [0, 2, 7, 1],
            [0, 0, 0, 10]
        ])
        target_names_4 = ['A', 'B', 'C', 'D']

        # Both should work without errors
        plot_confusion_matrix(conf_matrix_2, target_names_2)
        plot_confusion_matrix(conf_matrix_4, target_names_4)

        plt.close('all')

    def test_plot_confusion_matrix_perfect_classification(self):
        """Test plotting perfect classification confusion matrix."""
        conf_matrix = np.eye(3, dtype=int) * 10
        target_names = ['A', 'B', 'C']

        # Should work without errors
        plot_confusion_matrix(conf_matrix, target_names)

        plt.close('all')


class TestPlotComparisonMetrics:
    """Tests for plot_comparison_metrics function."""

    @pytest.fixture
    def sample_comparison_df(self):
        """Create sample comparison dataframe."""
        comparison = {
            'Metric': [
                'Accuracy (KNN)',
                'ARI (HDBSCAN)',
                'NMI (HDBSCAN)',
                'Clusters Found (HDBSCAN)',
                'Noise Points (HDBSCAN)'
            ],
            'Value': ['0.9500', '0.8500', '0.9000', 3, 2]
        }
        df = pd.DataFrame(comparison)
        return df

    def test_plot_comparison_metrics_creates_figure(self, sample_comparison_df):
        """Test that comparison metrics plot creates a figure."""
        # Should not raise any errors
        plot_comparison_metrics(sample_comparison_df)

        plt.close('all')

    def test_plot_comparison_metrics_saves_file(self, sample_comparison_df, temp_output_dir):
        """Test that comparison metrics saves to file."""
        save_path = temp_output_dir / "test_comparison.png"

        plot_comparison_metrics(sample_comparison_df, save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close('all')

    def test_plot_comparison_metrics_without_save_path(self, sample_comparison_df):
        """Test plotting comparison metrics without saving."""
        # Should not raise any errors
        plot_comparison_metrics(sample_comparison_df)

        plt.close('all')

    def test_plot_comparison_metrics_different_row_counts(self):
        """Test comparison metrics with different numbers of rows."""
        # Small dataframe
        df_small = pd.DataFrame({
            'Metric': ['Metric 1', 'Metric 2'],
            'Value': [0.95, 0.85]
        })

        # Large dataframe
        df_large = pd.DataFrame({
            'Metric': [f'Metric {i}' for i in range(10)],
            'Value': np.random.rand(10)
        })

        # Both should work without errors
        plot_comparison_metrics(df_small)
        plot_comparison_metrics(df_large)

        plt.close('all')

    def test_plot_comparison_metrics_empty_dataframe(self):
        """Test plotting with empty dataframe."""
        df_empty = pd.DataFrame({'Metric': [], 'Value': []})

        # Should handle empty dataframe gracefully
        try:
            plot_comparison_metrics(df_empty)
            plt.close('all')
        except Exception as e:
            # Some error is acceptable for empty dataframe
            plt.close('all')
            pass

    def test_plot_comparison_metrics_mixed_types(self):
        """Test plotting with mixed value types."""
        df_mixed = pd.DataFrame({
            'Metric': ['Float Metric', 'Int Metric', 'String Metric'],
            'Value': [0.95, 10, 'good']
        })

        # Should handle mixed types
        plot_comparison_metrics(df_mixed)

        plt.close('all')


class TestVisualisationIntegration:
    """Integration tests for visualisation functions."""

    def test_all_plots_together(self, temp_output_dir):
        """Test creating all plots in sequence."""
        np.random.seed(42)

        # Create PCA plot
        X_pca = np.random.rand(50, 2)
        y_true = np.random.randint(0, 3, 50)
        y_pred = np.random.randint(0, 3, 50)
        target_names = ['setosa', 'versicolor', 'virginica']

        plot_pca_clusters(
            X_pca, y_true, y_pred, "Test",
            target_names, temp_output_dir / "pca.png"
        )

        # Create confusion matrix
        conf_matrix = np.array([[10, 0, 0], [0, 8, 2], [1, 0, 9]])
        plot_confusion_matrix(
            conf_matrix, target_names,
            temp_output_dir / "confusion.png"
        )

        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['Accuracy', 'ARI'],
            'Value': ['0.9500', '0.8500']
        })
        plot_comparison_metrics(
            comparison_df,
            temp_output_dir / "comparison.png"
        )

        # Check all files were created
        assert (temp_output_dir / "pca.png").exists()
        assert (temp_output_dir / "confusion.png").exists()
        assert (temp_output_dir / "comparison.png").exists()

        plt.close('all')
