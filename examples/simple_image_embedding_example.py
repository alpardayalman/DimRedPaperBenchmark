#!/usr/bin/env python3
"""Simple example for using the dimension reduction toolkit with image embeddings.

This script shows the basic workflow for analyzing image embeddings
from any source (CNN, ViT, CLIP, etc.).
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dimension_reduction import DimensionReducer
from benchmarking import BenchmarkSuite
from utils.image_embedding_utils import ImageEmbeddingAnalyzer


def main():
    """Simple example with image embeddings."""
    print("Simple Image Embedding Analysis")
    print("="*40)
    
    # Example: Load your image embeddings here
    # Replace this with your actual embedding loading code
    
    # Option 1: Load from numpy file
    # embeddings = np.load('your_embeddings.npy')
    
    # Option 2: Load from CSV
    # import pandas as pd
    # embeddings = pd.read_csv('your_embeddings.csv').values
    
    # Option 3: Load from your model
    # embeddings = your_model.encode(images)
    
    # For this example, we'll generate synthetic embeddings
    print("Generating example image embeddings...")
    n_images = 500
    n_features = 512
    n_classes = 5
    
    # Generate synthetic embeddings with class structure
    np.random.seed(42)
    labels = np.random.randint(0, n_classes, n_images)
    embeddings = np.random.randn(n_images, n_features)
    
    # Add class-specific patterns
    for class_id in range(n_classes):
        class_mask = labels == class_id
        if np.sum(class_mask) > 0:
            class_bias = np.random.randn(n_features) * 0.5
            embeddings[class_mask] += class_bias
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Step 1: Basic embedding analysis
    print("\n1. Basic Embedding Analysis")
    print("-" * 30)
    
    analyzer = ImageEmbeddingAnalyzer(embeddings, normalize=True)
    
    # Get basic statistics
    stats = analyzer.analyze_embedding_statistics()
    print("Embedding Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Step 2: Apply dimension reduction
    print("\n2. Applying Dimension Reduction")
    print("-" * 30)
    
    reducer = DimensionReducer()
    methods = ['pca', 'umap', 'tsne']
    reduced_embeddings = {}
    
    for method in methods:
        print(f"Applying {method.upper()}...")
        try:
            if method == 'tsne':
                # Use max_iter instead of n_iter for newer scikit-learn versions
                reduced = reducer.fit_transform(embeddings, method=method, n_components=2)
            else:
                reduced = reducer.fit_transform(embeddings, method=method, n_components=2)
            reduced_embeddings[method.upper()] = reduced
            print(f"  {method.upper()} completed: {reduced.shape}")
        except Exception as e:
            print(f"  Error with {method}: {e}")
    
    # Step 3: Run comprehensive benchmark
    print("\n3. Running Benchmark")
    print("-" * 30)
    
    benchmark = BenchmarkSuite(random_state=42)
    results = benchmark.evaluate_all(embeddings, reduced_embeddings, y_labels=labels)
    
    # Step 4: Print results
    print("\n4. Results Summary")
    print("-" * 30)
    
    if 'summary' in results:
        summary = results['summary']
        
        print("Best Methods by Category:")
        for category, method in summary['best_methods'].items():
            print(f"  {category.replace('_', ' ').title()}: {method}")
        
        print("\nOverall Scores:")
        for method, score in summary['overall_scores'].items():
            print(f"  {method}: {score:.4f}")
    
    # Step 5: Generate detailed report
    print("\n5. Detailed Metrics")
    print("-" * 30)
    
    report_df = benchmark.generate_report(results)
    print(report_df.round(4))
    
    # Step 6: Create visualization
    print("\n6. Creating Visualization")
    print("-" * 30)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(reduced_embeddings), figsize=(5*len(reduced_embeddings), 4))
        
        if len(reduced_embeddings) == 1:
            axes = [axes]
        
        for i, (method, embedding) in enumerate(reduced_embeddings.items()):
            ax = axes[i]
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.7)
            ax.set_title(f'{method} Embeddings')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('image_embeddings_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualization saved as 'image_embeddings_visualization.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    
    print("\n" + "="*40)
    print("ANALYSIS COMPLETE!")
    print("="*40)
    
    print("\nNext steps:")
    print("1. Replace the synthetic embeddings with your real image embeddings")
    print("2. Adjust parameters based on your specific use case")
    print("3. Use the results in your research paper")
    print("4. Consider running additional analyses (clustering, classification, etc.)")


if __name__ == "__main__":
    main()
