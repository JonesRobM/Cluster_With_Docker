"""Evaluation metrics module."""
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


def evaluate_knn(y_true, y_pred, target_names=None):
    """
    Evaluate KNN classifier performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix
    }

    if target_names is not None:
        class_report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=True
        )
        metrics['classification_report'] = class_report

    return metrics


def evaluate_hdbscan(y_true, y_pred):
    """
    Evaluate HDBSCAN clustering performance.

    Args:
        y_true: True labels
        y_pred: Cluster labels

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Count noise points (-1 labels)
    n_noise = sum(y_pred == -1)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

    metrics = {
        'ari': ari,
        'nmi': nmi,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }

    return metrics


def create_comparison_table(knn_metrics, hdbscan_metrics):
    """
    Create a comparison table of metrics.

    Args:
        knn_metrics: KNN evaluation metrics
        hdbscan_metrics: HDBSCAN evaluation metrics

    Returns:
        df: Pandas DataFrame with comparison
    """
    comparison = {
        'Metric': [
            'Accuracy (KNN)',
            'ARI (HDBSCAN)',
            'NMI (HDBSCAN)',
            'Clusters Found (HDBSCAN)',
            'Noise Points (HDBSCAN)'
        ],
        'Value': [
            f"{knn_metrics['accuracy']:.4f}",
            f"{hdbscan_metrics['ari']:.4f}",
            f"{hdbscan_metrics['nmi']:.4f}",
            hdbscan_metrics['n_clusters'],
            hdbscan_metrics['n_noise']
        ]
    }

    df = pd.DataFrame(comparison)
    return df
