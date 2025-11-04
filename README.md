# Iris Classification Experiment: KNN vs HDBSCAN

Comparative analysis of supervised classification (K-Nearest Neighbors) and unsupervised clustering (HDBSCAN) on the Iris dataset.

## Overview

This project implements an experiment comparing:
- **K-Nearest Neighbors (KNN)**: Supervised classification
- **HDBSCAN**: Hierarchical density-based clustering (unsupervised)

Both models are evaluated on the classic Iris dataset with standardized features and PCA visualization.

## Project Structure

```
Cluster_With_Docker/
├── src/                        # Source modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── models.py              # KNN and HDBSCAN model wrappers
│   ├── metrics.py             # Evaluation metrics
│   └── visualization.py       # Plotting functions
├── outputs/                    # Generated results
│   ├── plots/                 # Visualizations (PCA, confusion matrix)
│   └── metrics/               # Metrics (JSON, CSV)
├── evaluate.py                # Main evaluation script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Docker Compose configuration
├── CLAUDE.md                  # Experiment design documentation
└── README.md                  # This file
```

## Quick Start with Docker

### Build and run the experiment:

```bash
docker-compose up --build
```

This will:
1. Build the Docker image with all dependencies
2. Run the evaluation script
3. Save results to `outputs/` directory

### Results

After running, check the `outputs/` directory:

**Plots** (`outputs/plots/`):
- `knn_clusters.png` - PCA visualization of KNN predictions
- `hdbscan_clusters.png` - PCA visualization of HDBSCAN clusters
- `knn_confusion_matrix.png` - KNN confusion matrix
- `metrics_comparison.png` - Comparison table visualization

**Metrics** (`outputs/metrics/`):
- `evaluation_metrics.json` - Detailed metrics in JSON format
- `comparison_table.csv` - Comparison table in CSV format

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the experiment

```bash
python evaluate.py
```

## Models

### K-Nearest Neighbors (KNN)
- **Type**: Supervised classifier
- **Parameters**: `n_neighbors=5`, `metric='euclidean'`
- **Metric**: Accuracy on test split

### HDBSCAN
- **Type**: Unsupervised clustering
- **Parameters**: `min_cluster_size=5`
- **Metrics**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)

## Dataset

- **Source**: `sklearn.datasets.load_iris`
- **Features**: 4 numeric (sepal length, sepal width, petal length, petal width)
- **Target**: 3 classes (setosa, versicolor, virginica)
- **Samples**: 150 total (120 train, 30 test for KNN; all used for HDBSCAN)

## License

See LICENSE file for details.
