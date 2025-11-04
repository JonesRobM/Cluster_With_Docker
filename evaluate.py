"""Main evaluation script for comparing KNN and HDBSCAN on Iris dataset."""
import os
import json
from pathlib import Path

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


def main():
    """Run the complete evaluation experiment."""
    print("=" * 60)
    print("Iris Classification Experiment: KNN vs HDBSCAN")
    print("=" * 60)

    # Create output directories
    output_dir = Path("outputs")
    plots_dir = output_dir / "plots"
    metrics_dir = output_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess data
    print("\n[1/6] Loading Iris dataset...")
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_preprocess_data(
        test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {feature_names}")
    print(f"  Classes: {list(target_names)}")

    # 2. Standardise features
    print("\n[2/6] Standardising features...")
    X_train_scaled, X_test_scaled, scaler = standardise_features(X_train, X_test)

    # For HDBSCAN, use the full dataset
    import numpy as np
    X_full = np.vstack([X_train, X_test])
    y_full = np.hstack([y_train, y_test])
    X_full_scaled, _ = standardise_features(X_full)

    # 3. Train and evaluate KNN
    print("\n[3/6] Training K-Nearest Neighbours...")
    knn = KNNModel(n_neighbours=5, metric='euclidean')
    knn.train(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)

    knn_metrics = evaluate_knn(y_test, y_pred_knn, target_names)
    print(f"  KNN Accuracy: {knn_metrics['accuracy']:.4f}")

    # 4. Run HDBSCAN
    print("\n[4/6] Running HDBSCAN clustering...")
    hdbscan_model = HDBSCANModel(min_cluster_size=5, min_samples=None)
    y_pred_hdbscan = hdbscan_model.fit_predict(X_full_scaled)

    hdbscan_metrics = evaluate_hdbscan(y_full, y_pred_hdbscan)
    print(f"  HDBSCAN ARI: {hdbscan_metrics['ari']:.4f}")
    print(f"  HDBSCAN NMI: {hdbscan_metrics['nmi']:.4f}")
    print(f"  Clusters found: {hdbscan_metrics['n_clusters']}")
    print(f"  Noise points: {hdbscan_metrics['n_noise']}")

    # 5. Generate visualisations
    print("\n[5/6] Generating visualisations...")

    # PCA for visualisation
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

    # 6. Create comparison table
    print("\n[6/6] Creating comparison metrics...")
    comparison_df = create_comparison_table(knn_metrics, hdbscan_metrics)
    print("\n" + comparison_df.to_string(index=False))

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
    print(f"\nMetrics saved to {metrics_file}")

    # Save comparison table as CSV
    comparison_file = metrics_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Comparison table saved to {comparison_file}")

    print("\n" + "=" * 60)
    print("Evaluation complete! Check the outputs/ directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
