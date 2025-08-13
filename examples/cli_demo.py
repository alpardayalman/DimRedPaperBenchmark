#!/usr/bin/env python3
"""CLI demo script for dimension reduction algorithms."""

import argparse
import sys
import numpy as np
from sklearn.datasets import make_swiss_roll, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from dimension_reduction.main import DimensionReducer


def generate_dataset(dataset_type: str, n_samples: int = 1000, noise: float = 0.1) -> np.ndarray:
    """Generate a test dataset.
    
    Args:
        dataset_type: Type of dataset to generate.
        n_samples: Number of samples.
        noise: Noise level.
        
    Returns:
        Generated dataset.
    """
    np.random.seed(42)
    
    if dataset_type == 'swiss_roll':
        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset_type == 'random':
        X = np.random.randn(n_samples, 20)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X


def plot_results(X_original: np.ndarray, results: dict, dataset_name: str) -> None:
    """Plot dimension reduction results.
    
    Args:
        X_original: Original data.
        results: Dictionary of results.
        dataset_name: Name of the dataset.
    """
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Dimension Reduction Results: {dataset_name.replace("_", " ").title()}', fontsize=16)
    
    # Plot original data
    ax = axes[0, 0]
    if X_original.shape[1] >= 2:
        ax.scatter(X_original[:, 0], X_original[:, 1], alpha=0.6, s=20)
    else:
        ax.scatter(X_original[:, 0], np.zeros_like(X_original[:, 0]), alpha=0.6, s=20)
    ax.set_title('Original Data')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Plot results for each method
    methods = ['pca', 'kernel_pca', 'isomap', 'lle', 'laplacian_eigenmap', 'tsne', 'umap', 'vae']
    
    for i, method in enumerate(methods):
        row = i // 4
        col = i % 4
        
        if method in results and results[method]['success']:
            X_transformed = results[method]['transformed_data']
            ax = axes[row, col]
            
            if X_transformed is not None and X_transformed.shape[1] >= 2:
                ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6, s=20)
                ax.set_title(f'{method.upper()}\nTime: {results[method]["elapsed_time"]:.2f}s')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
            else:
                ax.text(0.5, 0.5, f'{method.upper()}\n1D output', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method.upper()}')
        else:
            ax = axes[row, col]
            error_msg = results.get(method, {}).get('error', 'Failed')
            ax.text(0.5, 0.5, f'{method.upper()}\n{error_msg[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(f'{method.upper()}')
    
    plt.tight_layout()
    plt.show()


def print_summary(results: dict) -> None:
    """Print a summary of results.
    
    Args:
        results: Dictionary of results.
    """
    print("\n" + "="*80)
    print("DIMENSION REDUCTION RESULTS SUMMARY")
    print("="*80)
    
    # Create summary table
    summary_data = []
    
    for method, result in results.items():
        if result['success']:
            summary_data.append({
                'Method': method.upper(),
                'Success': '✓',
                'Time (s)': f"{result['elapsed_time']:.3f}",
                'Output Shape': str(result['transformed_data'].shape)
            })
        else:
            summary_data.append({
                'Method': method.upper(),
                'Success': '✗',
                'Time (s)': f"{result['elapsed_time']:.3f}",
                'Output Shape': 'N/A',
                'Error': result.get('error', 'Unknown error')
            })
    
    # Print summary table
    if summary_data:
        # Find the longest method name for formatting
        max_method_len = max(len(row['Method']) for row in summary_data)
        
        # Print header
        header = f"{'Method':<{max_method_len}} | {'Success':<8} | {'Time (s)':<10} | {'Output Shape':<15}"
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in summary_data:
            if 'Error' in row:
                print(f"{row['Method']:<{max_method_len}} | {row['Success']:<8} | {row['Time (s)']:<10} | {row['Output Shape']:<15}")
                print(f"{'':<{max_method_len}} | {'':<8} | {'':<10} | {'Error: ' + row['Error']}")
            else:
                print(f"{row['Method']:<{max_method_len}} | {row['Success']:<8} | {row['Time (s)']:<10} | {row['Output Shape']:<15}")
    
    print("="*80)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description='Dimension Reduction CLI Demo')
    parser.add_argument('--dataset', '-d', choices=['swiss_roll', 'moons', 'circles', 'random'], 
                       default='swiss_roll', help='Dataset type to use')
    parser.add_argument('--samples', '-n', type=int, default=1000, 
                       help='Number of samples')
    parser.add_argument('--noise', type=float, default=0.1, 
                       help='Noise level')
    parser.add_argument('--components', '-c', type=int, default=2, 
                       help='Number of components')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Skip plotting')
    parser.add_argument('--methods', '-m', nargs='+', 
                       choices=['pca', 'kernel_pca', 'isomap', 'lle', 'laplacian_eigenmap', 'tsne', 'umap', 'vae'],
                       default=['pca', 'kernel_pca', 'isomap', 'lle', 'laplacian_eigenmap', 'tsne', 'umap', 'vae'],
                       help='Methods to test')
    
    args = parser.parse_args()
    
    print("Dimension Reduction CLI Demo")
    print("="*40)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples}")
    print(f"Noise: {args.noise}")
    print(f"Components: {args.components}")
    print(f"Methods: {', '.join(args.methods)}")
    print()
    
    # Generate dataset
    print("Generating dataset...")
    X = generate_dataset(args.dataset, args.samples, args.noise)
    print(f"Dataset shape: {X.shape}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize dimension reducer
    reducer = DimensionReducer()
    
    # Test parameters for different algorithms
    test_params = {
        'kernel_pca': {'kernel': 'rbf', 'gamma': 0.1},
        'isomap': {'n_neighbors': 10},
        'lle': {'n_neighbors': 10, 'reg': 1e-3},
        'laplacian_eigenmap': {'n_neighbors': 10, 'affinity': 'nearest_neighbors'},
        'tsne': {'perplexity': 30, 'n_iter': 1000},
        'umap': {'n_neighbors': 15, 'min_dist': 0.1},
        'vae': {'epochs': 30, 'batch_size': 64, 'learning_rate': 1e-3}
    }
    
    # Evaluate methods
    results = {}
    for method in args.methods:
        if method in reducer.get_available_methods():
            print(f"Testing {method}...")
            params = test_params.get(method, {})
            
            try:
                start_time = time.time()
                X_transformed = reducer.fit_transform(X_scaled, method, args.components, **params)
                elapsed_time = time.time() - start_time
                
                results[method] = {
                    'success': True,
                    'transformed_data': X_transformed,
                    'elapsed_time': elapsed_time
                }
                print(f"  ✓ Success ({elapsed_time:.2f}s)")
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'elapsed_time': elapsed_time
                }
                print(f"  ✗ Failed: {e}")
        else:
            print(f"Method {method} not available, skipping...")
    
    # Print summary
    print_summary(results)
    
    # Plot results if requested
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_results(X, results, args.dataset)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    import time
    main()
