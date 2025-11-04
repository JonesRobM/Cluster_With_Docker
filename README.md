# Iris Classification Experiment: KNN vs HDBSCAN

Comparative analysis of supervised classification (K-Nearest Neighbours) and unsupervised clustering (HDBSCAN) on the Iris dataset.

## Overview

This project implements an experiment comparing:
- **K-Nearest Neighbours (KNN)**: Supervised classification
- **HDBSCAN**: Hierarchical density-based clustering (unsupervised)

Both models are evaluated on the classic Iris dataset with standardised features and PCA visualisation.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Cluster_With_Docker

# Install the package
pip install .

# OR for development (editable mode with tests)
pip install -e ".[dev]"
```

## Project Structure

```
Cluster_With_Docker/
├── src/                        # Source modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── models.py              # KNN and HDBSCAN model wrappers
│   ├── metrics.py             # Evaluation metrics
│   └── visualisation.py       # Plotting functions
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Shared test fixtures
│   ├── test_data_loader.py   # Tests for data loading
│   ├── test_models.py         # Tests for models
│   ├── test_metrics.py        # Tests for metrics
│   ├── test_visualisation.py  # Tests for visualisation
│   └── test_evaluate.py       # Integration tests
├── outputs/                    # Generated results
│   ├── plots/                 # Visualisations (PCA, confusion matrix)
│   └── metrics/               # Metrics (JSON, CSV)
├── evaluate.py                # Main evaluation script
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Package configuration and metadata
├── MANIFEST.in                # Package distribution manifest
├── pytest.ini                 # Pytest configuration
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
- `knn_clusters.png` - PCA visualisation of KNN predictions
- `hdbscan_clusters.png` - PCA visualisation of HDBSCAN clusters
- `knn_confusion_matrix.png` - KNN confusion matrix
- `metrics_comparison.png` - Comparison table visualisation

**Metrics** (`outputs/metrics/`):
- `evaluation_metrics.json` - Detailed metrics in JSON format
- `comparison_table.csv` - Comparison table in CSV format

## Expected Results

After running the evaluation, you should see the following results:

### Performance Metrics

```
Metric                      Value
-----------------------------------------
Accuracy (KNN)             ~0.93-0.97
ARI (HDBSCAN)              ~0.50-0.60
NMI (HDBSCAN)              ~0.65-0.75
Clusters Found (HDBSCAN)   2-3
Noise Points (HDBSCAN)     0-5
```

### Visualizations Explained

#### 1. KNN Clusters

![KNN clusters (PCA)](output/plots/knn-clusters.png)

**Left panel (True Labels):** Shows the actual iris species in PCA space
- Three distinct clusters representing setosa, versicolor, and virginica
- Setosa (typically purple/yellow) is well-separated from the other two species
- Versicolor and virginica have some overlap in the feature space

**Right panel (Predicted Clusters):** Shows KNN predictions on the test set
- Should closely match the true labels with high accuracy (~93-97%)
- Misclassifications typically occur at the boundary between versicolor and virginica
- The overlap region is where KNN may struggle

**What to look for:**
- Strong agreement between left and right panels
- Most prediction errors in the overlapping region between versicolor/virginica
- Clear separation of setosa from other species

#### 2. HDBSCAN Clusters (`hdbscan_clusters.png`)

**Left panel (True Labels):** Ground truth species labels

**Right panel (Predicted Clusters):** HDBSCAN unsupervised clustering results
- Typically finds 2-3 clusters
- Often groups versicolor and virginica together as a single cluster
- May identify some points as noise (shown in different color)
- Lower correspondence to true labels compared to KNN (expected for unsupervised)

**What to look for:**
- HDBSCAN successfully identifies setosa as a separate cluster
- Versicolor and virginica are often merged into one cluster due to their overlap
- Noise points (if any) typically appear in the overlap region
- This demonstrates the challenge of unsupervised learning without label information

#### 3. Confusion Matrix (`knn_confusion_matrix.png`)

Shows KNN classification performance across all three species:
- **Diagonal values** (top-left to bottom-right): Correct predictions
- **Off-diagonal values**: Misclassifications

**Expected pattern:**
- High values on the diagonal (10 correct predictions per class in test set)
- Setosa: Perfect or near-perfect classification (10/10)
- Versicolor/Virginica: Occasional confusion (1-2 misclassifications)
- Most errors between versicolor ↔ virginica (not setosa)

#### 4. Metrics Comparison (`metrics_comparison.png`)

Summary table comparing both approaches:
- **KNN Accuracy**: 93-97% - High performance with labeled training data
- **HDBSCAN ARI**: 50-60% - Moderate agreement with true labels
- **HDBSCAN NMI**: 65-75% - Moderate mutual information with true clustering
- **Clusters Found**: Shows HDBSCAN typically finds 2 clusters vs. 3 true classes
- **Noise Points**: Small number of points HDBSCAN couldn't assign to any cluster

**Key Insights:**
1. **Supervised vs Unsupervised**: KNN (supervised) significantly outperforms HDBSCAN (unsupervised) because it learns from labeled examples
2. **Cluster Detection**: HDBSCAN's 2-cluster result reflects the natural structure in the data (setosa vs. versicolor/virginica)
3. **Real-world Application**: Use KNN when labels are available; use HDBSCAN for exploratory analysis or when labels are unavailable

## Local Development

### Prerequisites
- Python 3.8+
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

# Install package with dependencies
pip install .

# OR install in editable mode for development
pip install -e .

# OR install with test dependencies
pip install -e ".[dev]"
```

### Run the experiment

```bash
python evaluate.py
```

### Run tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests with verbose output
pytest -v
```

## Models

### K-Nearest Neighbours (KNN)
- **Type**: Supervised classifier
- **Parameters**: `n_neighbours=5`, `metric='euclidean'`
- **Metric**: Accuracy on test split

### HDBSCAN
- **Type**: Unsupervised clustering
- **Parameters**: `min_cluster_size=5`
- **Metrics**: Adjusted Rand Index (ARI), Normalised Mutual Information (NMI)

## Dataset

- **Source**: `sklearn.datasets.load_iris`
- **Features**: 4 numeric (sepal length, sepal width, petal length, petal width)
- **Target**: 3 classes (setosa, versicolor, virginica)
- **Samples**: 150 total (120 train, 30 test for KNN; all used for HDBSCAN)

## License

See LICENSE file for details.
