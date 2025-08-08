#!/usr/bin/env python3
"""Command-line interface for the dimension reduction toolkit."""

import argparse
import sys
import os
import json
import time
from typing import Optional, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler

from dimension_reduction import DimensionReducer
from benchmarking import BenchmarkSuite


def load_data(data_type: str, n_samples: int = 1000, n_features: int = 50) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load or generate data for testing.
    
    Args:
        data_type: Type of data to load ('blobs', 'iris', 'random').
        n_samples: Number of samples for synthetic data.
        n_features: Number of features for synthetic data.
        
    Returns:
        Tuple of (X, y) where y may be None.
    """
    if data_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=5, random_state=42)
        return X, y
    elif data_type == 'iris':
        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data, iris.target
    elif data_type == 'random':
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 5, n_samples)
        return X, y
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def run_benchmark(
    X: np.ndarray,
    y: Optional[np.ndarray],
    methods: list[str],
    n_components: int,
    output_dir: str
) -> None:
    """Run dimension reduction benchmark.
    
    Args:
        X: Input data.
        y: Target labels.
        methods: List of methods to test.
        n_components: Number of components.
        output_dir: Output directory for results.
    """
    print(f"Running benchmark on data of shape {X.shape}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Target dimensions: {n_components}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimension reduction
    reducer = DimensionReducer()
    embeddings = {}
    
    for method in methods:
        print(f"\nApplying {method.upper()}...")
        start_time = time.time()
        
        try:
            if method == 'tsne':
                embedding = reducer.fit_transform(
                    X_scaled, method=method, n_components=n_components, max_iter=300
                )
            else:
                embedding = reducer.fit_transform(
                    X_scaled, method=method, n_components=n_components
                )
            
            embeddings[method.upper()] = embedding
            elapsed_time = time.time() - start_time
            print(f"  Completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"  Error applying {method}: {e}")
    
    if not embeddings:
        print("No embeddings generated. Exiting.")
        return
    
    # Run benchmark
    print("\nRunning comprehensive benchmark...")
    benchmark = BenchmarkSuite(random_state=42)
    results = benchmark.evaluate_all(X_scaled, embeddings, y_labels=y)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    benchmark.save_results(results, os.path.join(output_dir, 'benchmark_results.json'))
    
    # Save as CSV
    report_df = benchmark.generate_report(results)
    report_df.to_csv(os.path.join(output_dir, 'benchmark_results.csv'), index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if 'summary' in results:
        summary = results['summary']
        
        print("\nBest Methods by Category:")
        for category, method in summary['best_methods'].items():
            print(f"  {category.replace('_', ' ').title()}: {method}")
        
        print("\nOverall Scores:")
        for method, score in summary['overall_scores'].items():
            print(f"  {method}: {score:.4f}")
    
    print(f"\nResults saved to: {output_dir}")


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Dimension Reduction Toolkit - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on synthetic data
  python cli.py --data-type blobs --methods pca umap --output results/

  # Run benchmark on iris dataset
  python cli.py --data-type iris --methods pca tsne umap --output iris_results/

  # Run with custom parameters
  python cli.py --data-type random --n-samples 500 --n-features 30 --methods pca umap --output custom_results/
        """
    )
    
    parser.add_argument(
        '--data-type',
        choices=['blobs', 'iris', 'random'],
        default='blobs',
        help='Type of data to use for testing'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['pca', 'tsne', 'umap', 'autoencoder'],
        default=['pca', 'umap'],
        help='Dimension reduction methods to test'
    )
    
    parser.add_argument(
        '--n-components',
        type=int,
        default=2,
        help='Number of components for dimension reduction'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples for synthetic data'
    )
    
    parser.add_argument(
        '--n-features',
        type=int,
        default=50,
        help='Number of features for synthetic data'
    )
    
    parser.add_argument(
        '--output',
        default='benchmark_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Dimension Reduction Toolkit CLI")
        print("="*40)
        print(f"Data type: {args.data_type}")
        print(f"Methods: {', '.join(args.methods)}")
        print(f"Components: {args.n_components}")
        print(f"Output: {args.output}")
    
    try:
        # Load data
        X, y = load_data(args.data_type, args.n_samples, args.n_features)
        
        if args.verbose:
            print(f"Loaded data: {X.shape}")
            if y is not None:
                print(f"Classes: {len(np.unique(y))}")
        
        # Run benchmark
        run_benchmark(X, y, args.methods, args.n_components, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
