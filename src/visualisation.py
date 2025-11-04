"""Visualisation module for clustering results."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_pca_clusters(X_pca, y_true, y_pred, title, target_names=None, save_path=None):
    """
    Plot PCA projection with true labels vs predicted clusters.

    Args:
        X_pca: PCA-reduced features (2D)
        y_true: True labels
        y_pred: Predicted labels/clusters
        title: Plot title
        target_names: Names of target classes
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot true labels
    scatter1 = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_true, cmap='viridis',
        s=50, alpha=0.6, edgecolors='k'
    )
    axes[0].set_title('True Labels')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    if target_names is not None:
        legend1 = axes[0].legend(
            handles=scatter1.legend_elements()[0],
            labels=list(target_names),
            title="Classes"
        )
    else:
        plt.colorbar(scatter1, ax=axes[0])

    # Plot predicted clusters
    scatter2 = axes[1].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_pred, cmap='viridis',
        s=50, alpha=0.6, edgecolors='k'
    )
    axes[1].set_title(f'Predicted Clusters ({title})')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_confusion_matrix(conf_matrix, target_names, save_path=None):
    """
    Plot confusion matrix heatmap.

    Args:
        conf_matrix: Confusion matrix
        target_names: Names of target classes
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d',
        cmap='Blues', xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title('KNN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_comparison_metrics(comparison_df, save_path=None):
    """
    Plot comparison metrics as a table.

    Args:
        comparison_df: DataFrame with comparison metrics
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=comparison_df.values,
        colLabels=comparison_df.columns,
        cellLoc='left',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Model Comparison Metrics', fontsize=14, weight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison table saved to {save_path}")

    plt.close()
