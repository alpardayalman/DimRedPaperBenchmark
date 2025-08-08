#!/usr/bin/env python3
"""Comprehensive example for analyzing image embeddings.

This script demonstrates how to use the dimension reduction toolkit
specifically for image embeddings from various sources (CNN, ViT, CLIP, etc.).
"""

import sys
import os
import time
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from dimension_reduction import DimensionReducer
from benchmarking import BenchmarkSuite
from utils.image_embedding_utils import ImageEmbeddingAnalyzer, load_image_embeddings, save_image_embeddings


def generate_synthetic_image_embeddings(
    n_images: int = 1000,
    n_features: int = 512,
    n_classes: int = 10,
    embedding_type: str = 'cnn'
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic image embeddings for testing.
    
    Args:
        n_images: Number of images.
        n_features: Number of embedding features.
        n_classes: Number of classes.
        embedding_type: Type of embedding ('cnn', 'vit', 'clip').
        
    Returns:
        Tuple of (embeddings, labels).
    """
    # Generate class labels
    labels = np.random.randint(0, n_classes, n_images)
    
    # Generate embeddings with class structure
    embeddings = np.random.randn(n_images, n_features)
    
    # Add class-specific patterns
    for class_id in range(n_classes):
        class_mask = labels == class_id
        if np.sum(class_mask) > 0:
            # Add class-specific bias to some features
            class_bias = np.random.randn(n_features) * 0.5
            embeddings[class_mask] += class_bias
    
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    
    # Add embedding-type specific characteristics
    if embedding_type == 'cnn':
        # CNN embeddings often have some sparsity
        mask = np.random.rand(*embeddings.shape) < 0.1
        embeddings[mask] = 0
    elif embedding_type == 'vit':
        # ViT embeddings often have higher variance
        embeddings *= 1.2
    elif embedding_type == 'clip':
        # CLIP embeddings are often more normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
    
    return embeddings, labels


def analyze_single_embedding_source(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    source_name: str = "Image Embeddings"
) -> Dict[str, Any]:
    """Analyze a single embedding source.
    
    Args:
        embeddings: Image embeddings.
        labels: Optional labels.
        source_name: Name of the embedding source.
        
    Returns:
        Analysis results.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {source_name}")
    print(f"Shape: {embeddings.shape}")
    print('='*60)
    
    # Create analyzer
    analyzer = ImageEmbeddingAnalyzer(embeddings, normalize=True)
    
    # Basic statistics
    print("\n1. Basic Statistics:")
    stats = analyzer.analyze_embedding_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Semantic clustering
    print("\n2. Semantic Clustering:")
    clusters = analyzer.analyze_semantic_clusters(labels, n_clusters=min(10, len(np.unique(labels)) if labels is not None else 10))
    print(f"  Silhouette Score: {clusters['silhouette_score']:.4f}")
    
    if labels is not None and 'adjusted_rand_score' in clusters:
        print(f"  Adjusted Rand Score: {clusters['adjusted_rand_score']:.4f}")
    
    # Feature importance (if labels available)
    if labels is not None:
        print("\n3. Feature Importance:")
        feature_importance = analyzer.analyze_feature_importance(labels)
        top_features = np.argsort(feature_importance['feature_importance'])[-10:]
        print(f"  Top 10 most important features: {top_features}")
    
    # Create comprehensive report
    report = analyzer.create_embedding_report(labels)
    
    return report


def compare_embedding_sources(
    embeddings_dict: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compare multiple embedding sources.
    
    Args:
        embeddings_dict: Dictionary of embeddings from different sources.
        labels: Optional labels.
        
    Returns:
        Comparison results.
    """
    print(f"\n{'='*80}")
    print("COMPARING EMBEDDING SOURCES")
    print('='*80)
    
    # Run dimension reduction on all sources
    reducer = DimensionReducer()
    reduced_embeddings = {}
    
    for source_name, embeddings in embeddings_dict.items():
        print(f"\nApplying dimension reduction to {source_name}...")
        
        # Apply different methods
        methods = ['pca', 'umap']
        source_reduced = {}
        
        for method in methods:
            try:
                if method == 'tsne':
                    # Use fewer iterations for faster computation
                    reduced = reducer.fit_transform(
                        embeddings, method=method, n_components=2, max_iter=300
                    )
                else:
                    reduced = reducer.fit_transform(
                        embeddings, method=method, n_components=2
                    )
                source_reduced[method.upper()] = reduced
                print(f"  {method.upper()}: Completed")
            except Exception as e:
                print(f"  {method.upper()}: Error - {e}")
        
        reduced_embeddings[source_name] = source_reduced
    
    # Run comprehensive benchmark
    print("\nRunning comprehensive benchmark...")
    benchmark = BenchmarkSuite(random_state=42)
    
    # Create combined embeddings for benchmark
    all_embeddings = {}
    for source_name, source_reduced in reduced_embeddings.items():
        for method_name, reduced_emb in source_reduced.items():
            all_embeddings[f"{source_name}_{method_name}"] = reduced_emb
    
    # Use original embeddings for benchmark (not reduced ones)
    benchmark_embeddings = {}
    for source_name, embeddings in embeddings_dict.items():
        benchmark_embeddings[source_name] = embeddings
    
    results = benchmark.evaluate_all(
        list(embeddings_dict.values())[0],  # Use first embedding as reference
        benchmark_embeddings,
        y_labels=labels
    )
    
    return {
        'reduced_embeddings': reduced_embeddings,
        'benchmark_results': results
    }


def create_visualizations(
    reduced_embeddings: Dict[str, Dict[str, np.ndarray]],
    labels: Optional[np.ndarray] = None,
    output_dir: str = "image_embedding_results"
) -> None:
    """Create visualizations for all embedding sources.
    
    Args:
        reduced_embeddings: Dictionary of reduced embeddings.
        labels: Optional labels.
        output_dir: Output directory for plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plots
    for method in ['PCA', 'UMAP']:
        sources_with_method = [
            source for source, methods in reduced_embeddings.items()
            if method in methods
        ]
        
        if not sources_with_method:
            continue
        
        n_sources = len(sources_with_method)
        fig, axes = plt.subplots(1, n_sources, figsize=(6*n_sources, 5))
        
        if n_sources == 1:
            axes = [axes]
        
        for i, source_name in enumerate(sources_with_method):
            ax = axes[i]
            embedding = reduced_embeddings[source_name][method]
            
            if labels is not None:
                scatter = ax.scatter(
                    embedding[:, 0], embedding[:, 1],
                    c=labels, cmap='viridis', alpha=0.7, s=30
                )
                ax.set_title(f'{source_name} - {method}')
            else:
                ax.scatter(
                    embedding[:, 0], embedding[:, 1],
                    alpha=0.7, s=30
                )
                ax.set_title(f'{source_name} - {method}')
            
            ax.set_xlabel(f'{method} Component 1')
            ax.set_ylabel(f'{method} Component 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method.lower()}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def main():
    """Main function for image embedding analysis."""
    print("Image Embedding Analysis - Comprehensive Example")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic embeddings from different sources
    print("\n1. Generating synthetic image embeddings...")
    
    embeddings_dict = {}
    labels = None
    
    # Generate different types of embeddings
    embedding_types = ['cnn', 'vit', 'clip']
    for emb_type in embedding_types:
        embeddings, emb_labels = generate_synthetic_image_embeddings(
            n_images=1000,
            n_features=512,
            n_classes=10,
            embedding_type=emb_type
        )
        embeddings_dict[emb_type.upper()] = embeddings
        
        # Use labels from first embedding (they should be the same)
        if labels is None:
            labels = emb_labels
    
    print(f"Generated embeddings for: {list(embeddings_dict.keys())}")
    
    # Analyze each embedding source individually
    print("\n2. Individual embedding analysis...")
    individual_results = {}
    
    for source_name, embeddings in embeddings_dict.items():
        results = analyze_single_embedding_source(embeddings, labels, source_name)
        individual_results[source_name] = results
    
    # Compare embedding sources
    print("\n3. Comparing embedding sources...")
    comparison_results = compare_embedding_sources(embeddings_dict, labels)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    create_visualizations(comparison_results['reduced_embeddings'], labels)
    
    # Save results
    print("\n5. Saving results...")
    output_dir = "image_embedding_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    for source_name, embeddings in embeddings_dict.items():
        save_image_embeddings(
            embeddings, 
            os.path.join(output_dir, f'{source_name.lower()}_embeddings.npy')
        )
    
    # Save labels
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    
    # Save analysis results
    import json
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Save individual results
    with open(os.path.join(output_dir, 'individual_analysis.json'), 'w') as f:
        json.dump(convert_numpy(individual_results), f, indent=2)
    
    # Save benchmark results
    benchmark = BenchmarkSuite()
    benchmark.save_results(
        comparison_results['benchmark_results'],
        os.path.join(output_dir, 'benchmark_results.json')
    )
    
    # Generate and save report
    report_df = benchmark.generate_report(comparison_results['benchmark_results'])
    report_df.to_csv(os.path.join(output_dir, 'benchmark_report.csv'), index=False)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    # Print summary
    if 'summary' in comparison_results['benchmark_results']:
        summary = comparison_results['benchmark_results']['summary']
        
        print("\nBest Methods by Category:")
        for category, method in summary['best_methods'].items():
            print(f"  {category.replace('_', ' ').title()}: {method}")
        
        print("\nOverall Scores:")
        for method, score in summary['overall_scores'].items():
            print(f"  {method}: {score:.4f}")
    
    print(f"\nAll results saved to: {output_dir}/")
    print("\nFiles generated:")
    print("- *_embeddings.npy: Original embeddings")
    print("- labels.npy: Image labels")
    print("- individual_analysis.json: Individual embedding analysis")
    print("- benchmark_results.json: Comprehensive benchmark results")
    print("- benchmark_report.csv: Summary table")
    print("- *_comparison.png: Visualization plots")


if __name__ == "__main__":
    main()
