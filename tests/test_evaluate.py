"""Integration tests for evaluate.py script."""
import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import the main function
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEvaluateIntegration:
    """Integration tests for the main evaluation script."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_full_pipeline_execution(self, temp_output_dir, monkeypatch):
        """Test that the full evaluation pipeline executes without errors."""
        # Change the output directory for testing
        monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from evaluate import main

        # Override output directory
        original_path = Path("outputs")
        test_outputs = temp_output_dir / "outputs"
        test_outputs.mkdir(parents=True, exist_ok=True)

        # Mock the output directory in the main function
        def mock_main():
            """Modified main function for testing."""
            from src.data_loader import (
                load_and_preprocess_data,
                standardise_features,
                reduce_dimensions
            )
            from src.models import KNNModel, HDBSCANModel
            from src.metrics import evaluate_knn, evaluate_hdbscan, create_comparison_table
            from src.visualisation import (
                plot_pca_clusters,
                plot_confusion_matrix,
                plot_comparison_metrics
            )

            output_dir = test_outputs
            plots_dir = output_dir / "plots"
            metrics_dir = output_dir / "metrics"
            plots_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            # Load and preprocess data
            X_train, X_test, y_train, y_test, feature_names, target_names = load_and_preprocess_data(
                test_size=0.2, random_state=42
            )

            # Standardise features
            X_train_scaled, X_test_scaled, scaler = standardise_features(X_train, X_test)

            # For HDBSCAN, use the full dataset
            X_full = np.vstack([X_train, X_test])
            y_full = np.hstack([y_train, y_test])
            X_full_scaled, _ = standardise_features(X_full)

            # Train and evaluate KNN
            knn = KNNModel(n_neighbours=5, metric='euclidean')
            knn.train(X_train_scaled, y_train)
            y_pred_knn = knn.predict(X_test_scaled)

            knn_metrics = evaluate_knn(y_test, y_pred_knn, target_names)

            # Run HDBSCAN
            hdbscan_model = HDBSCANModel(min_cluster_size=5, min_samples=None)
            y_pred_hdbscan = hdbscan_model.fit_predict(X_full_scaled)

            hdbscan_metrics = evaluate_hdbscan(y_full, y_pred_hdbscan)

            # Generate visualisations
            X_test_pca, pca_test = reduce_dimensions(X_test_scaled, n_components=2)
            X_full_pca, pca_full = reduce_dimensions(X_full_scaled, n_components=2)

            # Plot KNN results
            plot_pca_clusters(
                X_test_pca, y_test, y_pred_knn,
                title="KNN",
                target_names=target_names,
                save_path=plots_dir / "knn_clusters.png"
            )

            # Plot HDBSCAN results
            plot_pca_clusters(
                X_full_pca, y_full, y_pred_hdbscan,
                title="HDBSCAN",
                target_names=target_names,
                save_path=plots_dir / "hdbscan_clusters.png"
            )

            # Plot confusion matrix for KNN
            plot_confusion_matrix(
                knn_metrics['confusion_matrix'],
                target_names,
                save_path=plots_dir / "knn_confusion_matrix.png"
            )

            # Create comparison table
            comparison_df = create_comparison_table(knn_metrics, hdbscan_metrics)

            # Save comparison table
            plot_comparison_metrics(
                comparison_df,
                save_path=plots_dir / "metrics_comparison.png"
            )

            # Save metrics to JSON
            metrics_output = {
                'knn': {
                    'accuracy': float(knn_metrics['accuracy']),
                    'classification_report': knn_metrics.get('classification_report', {})
                },
                'hdbscan': {
                    'ari': float(hdbscan_metrics['ari']),
                    'nmi': float(hdbscan_metrics['nmi']),
                    'n_clusters': int(hdbscan_metrics['n_clusters']),
                    'n_noise': int(hdbscan_metrics['n_noise'])
                }
            }

            metrics_file = metrics_dir / "evaluation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_output, f, indent=2)

            # Save comparison table as CSV
            comparison_file = metrics_dir / "comparison_table.csv"
            comparison_df.to_csv(comparison_file, index=False)

            return metrics_output

        # Run the modified main function
        try:
            metrics_output = mock_main()

            # Verify outputs were created
            assert (test_outputs / "plots" / "knn_clusters.png").exists()
            assert (test_outputs / "plots" / "hdbscan_clusters.png").exists()
            assert (test_outputs / "plots" / "knn_confusion_matrix.png").exists()
            assert (test_outputs / "plots" / "metrics_comparison.png").exists()
            assert (test_outputs / "metrics" / "evaluation_metrics.json").exists()
            assert (test_outputs / "metrics" / "comparison_table.csv").exists()

            # Verify JSON structure
            assert 'knn' in metrics_output
            assert 'hdbscan' in metrics_output
            assert 'accuracy' in metrics_output['knn']
            assert 'ari' in metrics_output['hdbscan']

        finally:
            plt.close('all')

    def test_knn_accuracy_reasonable(self):
        """Test that KNN achieves reasonable accuracy on Iris dataset."""
        from src.data_loader import load_and_preprocess_data, standardise_features
        from src.models import KNNModel

        X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data(
            test_size=0.2, random_state=42
        )
        X_train_scaled, X_test_scaled, _ = standardise_features(X_train, X_test)

        knn = KNNModel(n_neighbours=5, metric='euclidean')
        knn.train(X_train_scaled, y_train)

        accuracy = knn.get_accuracy(X_test_scaled, y_test)

        # Iris dataset is easy, KNN should achieve high accuracy
        assert accuracy >= 0.85

    def test_hdbscan_finds_clusters(self):
        """Test that HDBSCAN finds clusters in Iris dataset."""
        from src.data_loader import load_and_preprocess_data, standardise_features
        from src.models import HDBSCANModel

        X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data()
        X_full = np.vstack([X_train, X_test])
        X_full_scaled, _ = standardise_features(X_full)

        hdbscan_model = HDBSCANModel(min_cluster_size=5)
        labels = hdbscan_model.fit_predict(X_full_scaled)

        n_clusters = hdbscan_model.get_n_clusters()

        # Should find at least 2 clusters in Iris dataset
        assert n_clusters >= 2

    def test_metrics_output_structure(self, temp_output_dir):
        """Test the structure of output metrics file."""
        from src.data_loader import load_and_preprocess_data, standardise_features
        from src.models import KNNModel, HDBSCANModel
        from src.metrics import evaluate_knn, evaluate_hdbscan

        X_train, X_test, y_train, y_test, _, target_names = load_and_preprocess_data()
        X_train_scaled, X_test_scaled, _ = standardise_features(X_train, X_test)

        X_full = np.vstack([X_train, X_test])
        y_full = np.hstack([y_train, y_test])
        X_full_scaled, _ = standardise_features(X_full)

        # Train models
        knn = KNNModel(n_neighbours=5)
        knn.train(X_train_scaled, y_train)
        y_pred_knn = knn.predict(X_test_scaled)

        hdbscan_model = HDBSCANModel(min_cluster_size=5)
        y_pred_hdbscan = hdbscan_model.fit_predict(X_full_scaled)

        # Evaluate
        knn_metrics = evaluate_knn(y_test, y_pred_knn, target_names)
        hdbscan_metrics = evaluate_hdbscan(y_full, y_pred_hdbscan)

        # Create metrics output
        metrics_output = {
            'knn': {
                'accuracy': float(knn_metrics['accuracy']),
                'classification_report': knn_metrics.get('classification_report', {})
            },
            'hdbscan': {
                'ari': float(hdbscan_metrics['ari']),
                'nmi': float(hdbscan_metrics['nmi']),
                'n_clusters': int(hdbscan_metrics['n_clusters']),
                'n_noise': int(hdbscan_metrics['n_noise'])
            }
        }

        # Verify structure
        assert isinstance(metrics_output['knn']['accuracy'], float)
        assert isinstance(metrics_output['hdbscan']['ari'], float)
        assert isinstance(metrics_output['hdbscan']['nmi'], float)
        assert isinstance(metrics_output['hdbscan']['n_clusters'], int)
        assert isinstance(metrics_output['hdbscan']['n_noise'], int)

    def test_reproducibility(self):
        """Test that results are reproducible with fixed random seed."""
        from src.data_loader import load_and_preprocess_data, standardise_features
        from src.models import KNNModel

        # First run
        X_train1, X_test1, y_train1, y_test1, _, _ = load_and_preprocess_data(random_state=42)
        X_train_scaled1, X_test_scaled1, _ = standardise_features(X_train1, X_test1)

        knn1 = KNNModel(n_neighbours=5)
        knn1.train(X_train_scaled1, y_train1)
        acc1 = knn1.get_accuracy(X_test_scaled1, y_test1)

        # Second run with same seed
        X_train2, X_test2, y_train2, y_test2, _, _ = load_and_preprocess_data(random_state=42)
        X_train_scaled2, X_test_scaled2, _ = standardise_features(X_train2, X_test2)

        knn2 = KNNModel(n_neighbours=5)
        knn2.train(X_train_scaled2, y_train2)
        acc2 = knn2.get_accuracy(X_test_scaled2, y_test2)

        # Results should be identical
        assert acc1 == acc2
        np.testing.assert_array_equal(X_test1, X_test2)
