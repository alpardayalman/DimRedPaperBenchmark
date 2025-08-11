# Dimension Reduction Toolkit

A comprehensive Python toolkit for dimension reduction techniques including t-SNE, PCA, and UMAP, with extensive benchmarking capabilities for research papers.

## Features

- **Multiple Dimension Reduction Algorithms**:
  - Principal Component Analysis (PCA)
  - Kernel PCA
  - Sparse PCA
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Uniform Manifold Approximation and Projection (UMAP)
  - Autoencoder-based dimension reduction
  - more to add

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
│   ├── tsne.py              # t-SNE implementation
│   ├── umap.py              # UMAP implementation
│   ├── autoencoder.py       # Autoencoder implementation
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:


