# Dimension Reduction Toolkit

A comprehensive Python toolkit for dimension reduction techniques including t-SNE, PCA, and UMAP, with extensive benchmarking capabilities for research papers.

## Citation

"This repository contains the code relative to [paper]. Please cite [paper] when referring to this repository."

## Features

- **Multiple Dimension Reduction Algorithms**:
  - **Linear Methods**:
    - Principal Component Analysis (PCA)
    - Kernel PCA (RBF, polynomial, sigmoid kernels)
  - **Manifold Learning**:
    - t-Distributed Stochastic Neighbor Embedding (t-SNE)
    - Uniform Manifold Approximation and Projection (UMAP)
    - ISOMAP (Isometric Mapping)
    - Locally Linear Embedding (LLE)
    - Laplacian Eigenmaps
  - **Deep Learning**:
    - Autoencoder-based dimension reduction
    - Variational Autoencoder (VAE)

- **Comprehensive Benchmarking**:
  - Preservation of local and global structure
  - Clustering quality assessment
  - Classification performance
  - Visualization quality metrics
  - Computational efficiency analysis

- **Research-Ready**:
  - Reproducible experiments
  - Statistical significance testing
  - Detailed performance metrics
  - Publication-ready visualizations

## Installation

```bash
# Install with uv (recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt


## Quick Start

```python
from src.dimension_reduction import DimensionReducer
from src.benchmarking import BenchmarkSuite
import numpy as np

# Generate sample data
X = np.random.randn(1000, 100)

# Initialize dimension reducer
reducer = DimensionReducer()

# Apply different techniques
pca_result = reducer.fit_transform(X, method='pca', n_components=2)
tsne_result = reducer.fit_transform(X, method='tsne', n_components=2)
umap_result = reducer.fit_transform(X, method='umap', n_components=2)

# Benchmark the results
benchmark = BenchmarkSuite()
results = benchmark.evaluate_all(X, {
    'PCA': pca_result,
    't-SNE': tsne_result,
    'UMAP': umap_result
})

print(results.summary())
```

## Benchmarking Strategies

### 1. Structure Preservation Metrics

- **Local Structure**: Neighborhood preservation, trustworthiness
- **Global Structure**: Continuity, stress, Shepard diagrams
- **Topological Structure**: Persistent homology, Mapper algorithm

### 2. Downstream Task Performance

- **Clustering**: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index
- **Classification**: Accuracy, F1-score, ROC-AUC with cross-validation
- **Regression**: R² score, mean squared error

### 3. Visualization Quality

- **Separation**: Inter-class distance, intra-class compactness
- **Overlap**: Jaccard similarity, overlap coefficient
- **Aesthetics**: Stress minimization, edge crossing reduction

### 4. Computational Efficiency

- **Time Complexity**: Training and inference time
- **Memory Usage**: Peak memory consumption
- **Scalability**: Performance vs. dataset size

## Project Structure

```
src/
├── dimension_reduction/
│   ├── __init__.py
│   ├── base.py              # Base classes and interfaces
│   ├── pca.py               # PCA implementation
│   ├── kernel_pca.py        # Kernel PCA implementation
│   ├── tsne.py              # t-SNE implementation
│   ├── umap.py              # UMAP implementation
│   ├── isomap.py            # ISOMAP implementation
│   ├── lle.py               # Locally Linear Embedding
│   ├── laplacian_eigenmap.py # Laplacian Eigenmaps
│   ├── autoencoder.py       # Autoencoder implementation
│   ├── vae.py               # Variational Autoencoder
│   └── cli.py               # Command-line interface
├── benchmarking/
│   ├── __init__.py
│   ├── metrics.py           # Evaluation metrics
│   ├── structure.py         # Structure preservation metrics
│   ├── clustering.py        # Clustering evaluation
│   ├── classification.py    # Classification evaluation
│   └── visualization.py     # Visualization quality metrics
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading utilities
│   ├── visualization.py     # Plotting utilities
│   └── reporting.py         # Results reporting
└── experiments/
    ├── __init__.py
    ├── benchmark_experiment.py  # Main benchmarking experiment
    └── hyperparameter_tuning.py # Hyperparameter optimization

tests/
├── __init__.py
├── test_dimension_reduction/
├── test_benchmarking/
└── test_utils/

docs/
├── api.md
├── benchmarking_guide.md
└── examples/
```

## Usage Examples

### Basic Dimension Reduction

```python
from src.dimension_reduction import DimensionReducer

# Initialize with default parameters
reducer = DimensionReducer()

# Fit and transform data
X_reduced = reducer.fit_transform(X, method='umap', n_components=2)
```

### Comprehensive Benchmarking

```python
from src.benchmarking import BenchmarkSuite
from src.dimension_reduction import DimensionReducer

# Prepare data and methods
methods = ['pca', 'tsne', 'umap']
reducer = DimensionReducer()

# Generate embeddings
embeddings = {}
for method in methods:
    embeddings[method] = reducer.fit_transform(X, method=method, n_components=2)

# Run comprehensive benchmark
benchmark = BenchmarkSuite()
results = benchmark.evaluate_all(X, embeddings, y_labels=y)

# Generate report
report = benchmark.generate_report(results)
report.save('benchmark_results.html')
```

### Custom Benchmarking

```python
from src.benchmarking import StructurePreservation, ClusteringQuality

# Custom structure preservation analysis
structure_metrics = StructurePreservation()
local_quality = structure_metrics.local_structure_preservation(X, embeddings)
global_quality = structure_metrics.global_structure_preservation(X, embeddings)

# Custom clustering analysis
clustering_metrics = ClusteringQuality()
silhouette_scores = clustering_metrics.silhouette_analysis(embeddings, y)
```

### Advanced Dimension Reduction Methods

The toolkit now includes several advanced dimension reduction algorithms:

#### Kernel PCA
```python
# Non-linear PCA using different kernel functions
kpca_rbf = reducer.fit_transform(X, method='kernel_pca', kernel='rbf', gamma=0.1)
kpca_poly = reducer.fit_transform(X, method='kernel_pca', kernel='poly', degree=3)
```

#### Manifold Learning
```python
# ISOMAP - preserves geodesic distances
isomap_result = reducer.fit_transform(X, method='isomap', n_neighbors=10)

# Locally Linear Embedding - preserves local neighborhood relationships
lle_result = reducer.fit_transform(X, method='lle', n_neighbors=10, reg=1e-3)

# Laplacian Eigenmaps - spectral embedding based on graph Laplacian
le_result = reducer.fit_transform(X, method='laplacian_eigenmap', n_neighbors=10)
```

#### Deep Learning Methods
```python
# Variational Autoencoder with custom architecture
vae_result = reducer.fit_transform(
    X, 
    method='vae', 
    hidden_dims=[128, 64], 
    epochs=100, 
    batch_size=32
)

# Autoencoder with custom parameters
ae_result = reducer.fit_transform(
    X, 
    method='autoencoder', 
    hidden_dims=[128, 64], 
    epochs=100
)
```

### Algorithm Comparison

```python
# Compare all available methods
methods = ['pca', 'kernel_pca', 'isomap', 'lle', 'laplacian_eigenmap', 'tsne', 'umap', 'vae']
results = reducer.compare_methods(X, methods=methods, n_components=2)

# Get detailed comparison with custom parameters
method_params = {
    'kernel_pca': {'kernel': 'rbf', 'gamma': 0.1},
    'isomap': {'n_neighbors': 15},
    'lle': {'n_neighbors': 15, 'reg': 1e-2},
    'laplacian_eigenmap': {'n_neighbors': 15, 'affinity': 'rbf'},
    'tsne': {'perplexity': 30, 'n_iter': 1000},
    'umap': {'n_neighbors': 15, 'min_dist': 0.1},
    'vae': {'epochs': 50, 'batch_size': 64}
}

detailed_results = reducer.compare_methods(
    X, 
    methods=methods, 
    n_components=2, 
    method_params=method_params
)
```

If you use this toolkit in your research, please cite:


