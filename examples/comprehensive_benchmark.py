#!/usr/bin/env python3
"""Comprehensive example demonstrating the dimension reduction toolkit.

This script shows how to use the dimension reduction algorithms and
benchmarking framework for a complete evaluation workflow.
"""

import sys
import os
import time
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_digits
from sklearn.preprocessing import StandardScaler

from dimension_reduction import DimensionReducer
from benchmarking import BenchmarkSuite


def generate_synthetic_datasets() -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic datasets for testing.
    
    Returns:
        Dictionary mapping dataset names to (X, y) tuples.
    """
    datasets = {}
    
    # Blobs dataset (well-separated clusters)
    X_blobs, y_blobs = make_blobs(
        n_samples=1000, n_features=50, centers=5, 
        cluster_std=1.0, random_state=42
    )
    datasets['blobs'] = (X_blobs, y_blobs)
    
    # Moons dataset (non-linear structure)
    X_moons, y_moons = make_moons(
        n_samples=1000, noise=0.1, random_state=42
    )
    # Add extra dimensions to make it high-dimensional
    X_moons_high = np.random.randn(1000, 48)
    X_moons_high = np.column_stack([X_moons, X_moons_high])
    datasets['moons'] = (X_moons_high, y_moons)
    
    # Circles dataset (concentric circles)
    X_circles, y_circles = make_circles(
        n_samples=1000, noise=0.1, factor=0.5, random_state=42
    )
    # Add extra dimensions
    X_circles_high = np.random.randn(1000, 48)
    X_circles_high = np.column_stack([X_circles, X_circles_high])
    datasets['circles'] = (X_circles_high, y_circles)
    
    # Random high-dimensional data
    X_random = np.random.randn(1000, 50)
    y_random = np.random.randint(0, 5, 1000)
    datasets['random'] = (X_random, y_random)
    
    return datasets


def load_real_datasets() -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load real datasets for testing.
    
    Returns:
        Dictionary mapping dataset names to (X, y) tuples.
    """
    datasets = {}
    
    # Iris dataset
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    datasets['iris'] = (X_iris, y_iris)
    
    # Digits dataset (subset for faster computation)
    digits = load_digits()
    X_digits = digits.data[:500]  # Use subset
    y_digits = digits.target[:500]
    datasets['digits'] = (X_digits, y_digits)
    
    return datasets


def apply_dimension_reduction(
    X: np.ndarray,
    methods: list[str] = None
) -> Dict[str, np.ndarray]:
    """Apply dimension reduction methods to the data.
    
    Args:
        X: Input data.
        methods: List of methods to apply.
        
    Returns:
        Dictionary mapping method names to embeddings.
    """
    if methods is None:
        methods = ['pca', 'tsne', 'umap']
    
    reducer = DimensionReducer()
    embeddings = {}
    
    print(f"Applying dimension reduction to data of shape {X.shape}")
    
    for method in methods:
        print(f"  Applying {method.upper()}...")
        start_time = time.time()
        
        try:
            if method == 'tsne':
                # Use fewer iterations for faster computation
                embedding = reducer.fit_transform(
                    X, method=method, n_components=2, n_iter=300
                )
            elif method == 'umap':
                embedding = reducer.fit_transform(
                    X, method=method, n_components=2, n_neighbors=15
                )
            else:
                embedding = reducer.fit_transform(
                    X, method=method, n_components=2
                )
            
            embeddings[method.upper()] = embedding
            elapsed_time = time.time() - start_time
            print(f"    Completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"    Error applying {method}: {e}")
    
    return embeddings


def run_benchmark(
    X: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    y: np.ndarray = None
) -> Dict[str, Any]:
    """Run comprehensive benchmarking.
    
    Args:
        X: Original data.
        embeddings: Dictionary of embeddings.
        y: Target labels.
        
    Returns:
        Benchmark results.
    """
    print("Running comprehensive benchmark...")
    
    benchmark = BenchmarkSuite(random_state=42)
    results = benchmark.evaluate_all(X, embeddings, y_labels=y)
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print benchmark summary.
    
    Args:
        results: Benchmark results.
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    if 'summary' in results:
        summary = results['summary']
        
        # Best methods
        print("\nBest Methods by Category:")
        for category, method in summary['best_methods'].items():
            print(f"  {category.replace('_', ' ').title()}: {method}")
        
        # Rankings
        print("\nMethod Rankings:")
        for category, ranking in summary['rankings'].items():
            print(f"  {category.replace('_', ' ').title()}:")
            for i, method in enumerate(ranking, 1):
                print(f"    {i}. {method}")
        
        # Overall scores
        print("\nOverall Scores:")
        for method, score in summary['overall_scores'].items():
            print(f"  {method}: {score:.4f}")
    
    # Generate detailed report
    benchmark = BenchmarkSuite()
    report_df = benchmark.generate_report(results)
    print("\nDetailed Metrics:")
    print(report_df.round(4))


def create_visualizations(
    embeddings: Dict[str, np.ndarray],
    y: np.ndarray,
    dataset_name: str
) -> None:
    """Create visualization plots.
    
    Args:
        embeddings: Dictionary of embeddings.
        y: Target labels.
        dataset_name: Name of the dataset.
    """
    n_methods = len(embeddings)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for i, (method, embedding) in enumerate(embeddings.items()):
        ax = axes[i]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax.set_title(f'{method} on {dataset_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved as visualizations_{dataset_name}.png")


def save_results(results: Dict[str, Any], dataset_name: str) -> None:
    """Save benchmark results.
    
    Args:
        results: Benchmark results.
        dataset_name: Name of the dataset.
    """
    # Save as JSON
    benchmark = BenchmarkSuite()
    benchmark.save_results(results, f'results_{dataset_name}.json')
    
    # Save as CSV
    report_df = benchmark.generate_report(results)
    report_df.to_csv(f'results_{dataset_name}.csv', index=False)
    
    print(f"Results saved as results_{dataset_name}.json and results_{dataset_name}.csv")


def main():
    """Main function demonstrating the toolkit."""
    print("Dimension Reduction Toolkit - Comprehensive Example")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate and load datasets
    print("\n1. Loading datasets...")
    synthetic_datasets = generate_synthetic_datasets()
    real_datasets = load_real_datasets()
    all_datasets = {**synthetic_datasets, **real_datasets}
    
    # Process each dataset
    for dataset_name, (X, y) in all_datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"Data shape: {X.shape}, Classes: {len(np.unique(y))}")
        print('='*60)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimension reduction
        print("\n2. Applying dimension reduction...")
        embeddings = apply_dimension_reduction(X_scaled)
        
        if not embeddings:
            print("No embeddings generated, skipping...")
            continue
        
        # Run benchmark
        print("\n3. Running benchmark...")
        results = run_benchmark(X_scaled, embeddings, y)
        
        # Print summary
        print_summary(results)
        
        # Create visualizations
        print("\n4. Creating visualizations...")
        create_visualizations(embeddings, y, dataset_name)
        
        # Save results
        print("\n5. Saving results...")
        save_results(results, dataset_name)
    
    print(f"\n{'='*60}")
    print("COMPLETE! All datasets processed successfully.")
    print("="*60)
    
    # Print final summary
    print("\nFinal Summary:")
    print("- Generated embeddings for all dimension reduction methods")
    print("- Evaluated using comprehensive benchmarking metrics")
    print("- Created visualizations for each dataset")
    print("- Saved detailed results and reports")
    print("\nFiles generated:")
    print("- results_*.json: Detailed benchmark results")
    print("- results_*.csv: Summary tables")
    print("- visualizations_*.png: Embedding plots")


if __name__ == "__main__":
    main()
