"""Comprehensive example demonstrating all dimension reduction algorithms."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_swiss_roll, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time

# Import our dimension reduction algorithms
from src.dimension_reduction.main import DimensionReducer

# Set random seed for reproducibility
np.random.seed(42)

def generate_test_datasets() -> dict:
    """Generate various test datasets for dimension reduction.
    
    Returns:
        Dictionary containing different test datasets.
    """
    datasets = {}
    
    # Swiss roll dataset (3D manifold)
    X_swiss, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    datasets['swiss_roll'] = X_swiss
    
    # Moons dataset (2D non-linear)
    X_moons, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
    datasets['moons'] = X_moons
    
    # Circles dataset (2D non-linear)
    X_circles, _ = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    datasets['circles'] = X_circles
    
    # High-dimensional random data
    X_random = np.random.randn(1000, 50)
    datasets['random'] = X_random
    
    return datasets

def evaluate_algorithm(
    reducer: DimensionReducer,
    X: np.ndarray,
    method: str,
    n_components: int = 2,
    **kwargs
) -> dict:
    """Evaluate a dimension reduction algorithm.
    
    Args:
        reducer: Dimension reducer instance.
        X: Input data.
        method: Method name.
        n_components: Number of components.
        **kwargs: Additional method parameters.
        
    Returns:
        Dictionary with evaluation results.
    """
    start_time = time.time()
    
    try:
        # Apply dimension reduction
        X_transformed = reducer.fit_transform(X, method, n_components, **kwargs)
        
        # Calculate silhouette score if we have enough components
        if n_components >= 2:
            try:
                # Use k-means clustering for silhouette score
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(5, X_transformed.shape[0]//10), random_state=42)
                labels = kmeans.fit_predict(X_transformed)
                silhouette = silhouette_score(X_transformed, labels)
            except:
                silhouette = np.nan
        else:
            silhouette = np.nan
        
        # Calculate reconstruction error if inverse transform is available
        try:
            model = reducer.get_model(method)
            if hasattr(model, 'inverse_transform'):
                X_reconstructed = model.inverse_transform(X_transformed)
                reconstruction_error = np.mean((X - X_reconstructed) ** 2)
            else:
                reconstruction_error = np.nan
        except:
            reconstruction_error = np.nan
        
        elapsed_time = time.time() - start_time
        
        return {
            'method': method,
            'n_components': n_components,
            'transformed_data': X_transformed,
            'silhouette_score': silhouette,
            'reconstruction_error': reconstruction_error,
            'elapsed_time': elapsed_time,
            'success': True
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'method': method,
            'n_components': n_components,
            'transformed_data': None,
            'silhouette_score': np.nan,
            'reconstruction_error': np.nan,
            'elapsed_time': elapsed_time,
            'success': False,
            'error': str(e)
        }

def plot_results(datasets: dict, results: dict, dataset_name: str) -> None:
    """Plot dimension reduction results.
    
    Args:
        datasets: Dictionary of datasets.
        results: Dictionary of results.
        dataset_name: Name of the dataset.
    """
    X_original = datasets[dataset_name]
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Dimension Reduction Results: {dataset_name.replace("_", " ").title()}', fontsize=16)
    
    # Plot original data (first 2 dimensions if available)
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
        row = (i + 1) // 4
        col = (i + 1) % 4
        
        if method in results and results[method]['success']:
            X_transformed = results[method]['transformed_data']
            ax = axes[row, col]
            
            if X_transformed is not None and X_transformed.shape[1] >= 2:
                ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6, s=20)
                ax.set_title(f'{method.upper()}\nSilhouette: {results[method]["silhouette_score"]:.3f}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
            else:
                ax.text(0.5, 0.5, f'{method.upper()}\n1D output', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method.upper()}')
        else:
            ax = axes[row, col]
            ax.text(0.5, 0.5, f'{method.upper()}\nFailed', 
                   ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(f'{method.upper()}')
    
    # Remove empty subplots
    for i in range(len(methods) + 1, 12):
        row = i // 4
        col = i % 4
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()

def print_evaluation_summary(results: dict) -> None:
    """Print a summary of evaluation results.
    
    Args:
        results: Dictionary of results.
    """
    print("\n" + "="*80)
    print("DIMENSION REDUCTION EVALUATION SUMMARY")
    print("="*80)
    
    # Create summary table
    summary_data = []
    
    for method, result in results.items():
        if result['success']:
            summary_data.append({
                'Method': method.upper(),
                'Success': '✓',
                'Silhouette Score': f"{result['silhouette_score']:.4f}" if not np.isnan(result['silhouette_score']) else 'N/A',
                'Reconstruction Error': f"{result['reconstruction_error']:.4f}" if not np.isnan(result['reconstruction_error']) else 'N/A',
                'Time (s)': f"{result['elapsed_time']:.3f}"
            })
        else:
            summary_data.append({
                'Method': method.upper(),
                'Success': '✗',
                'Silhouette Score': 'N/A',
                'Reconstruction Error': 'N/A',
                'Time (s)': f"{result['elapsed_time']:.3f}",
                'Error': result.get('error', 'Unknown error')
            })
    
    # Print summary table
    if summary_data:
        # Find the longest method name for formatting
        max_method_len = max(len(row['Method']) for row in summary_data)
        
        # Print header
        header = f"{'Method':<{max_method_len}} | {'Success':<8} | {'Silhouette':<12} | {'Reconstruction':<15} | {'Time (s)':<10}"
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in summary_data:
            if 'Error' in row:
                print(f"{row['Method']:<{max_method_len}} | {row['Success']:<8} | {'N/A':<12} | {'N/A':<15} | {row['Time (s)']:<10}")
                print(f"{'':<{max_method_len}} | {'':<8} | {'':<12} | {'':<15} | {'Error: ' + row['Error']}")
            else:
                print(f"{row['Method']:<{max_method_len}} | {row['Success']:<8} | {row['Silhouette Score']:<12} | {row['Reconstruction Error']:<15} | {row['Time (s)']:<10}")
    
    print("="*80)

def main() -> None:
    """Main function to run the comprehensive dimension reduction example."""
    print("Comprehensive Dimension Reduction Example")
    print("="*50)
    
    # Generate test datasets
    print("Generating test datasets...")
    datasets = generate_test_datasets()
    
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
        'vae': {'epochs': 50, 'batch_size': 64, 'learning_rate': 1e-3}
    }
    
    # Evaluate each dataset
    for dataset_name, X in datasets.items():
        print(f"\nEvaluating {dataset_name} dataset (shape: {X.shape})...")
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Evaluate all methods
        results = {}
        methods = ['pca', 'kernel_pca', 'isomap', 'lle', 'laplacian_eigenmap', 'tsne', 'umap', 'vae']
        
        for method in methods:
            print(f"  Testing {method}...")
            params = test_params.get(method, {})
            results[method] = evaluate_algorithm(reducer, X_scaled, method, 2, **params)
        
        # Print evaluation summary
        print_evaluation_summary(results)
        
        # Plot results
        plot_results(datasets, results, dataset_name)
        
        print(f"\nCompleted evaluation of {dataset_name} dataset.")

if __name__ == "__main__":
    main()
