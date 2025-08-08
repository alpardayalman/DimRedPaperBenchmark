#!/usr/bin/env python3
"""Test script to verify UMAP and t-SNE fixes."""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dimension_reduction import DimensionReducer

def test_fixes():
    """Test that UMAP and t-SNE work correctly."""
    print("Testing UMAP and t-SNE fixes...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    
    # Test UMAP
    print("\n1. Testing UMAP...")
    try:
        reducer = DimensionReducer()
        umap_result = reducer.fit_transform(X, method='umap', n_components=2)
        print(f"   UMAP successful! Shape: {umap_result.shape}")
    except Exception as e:
        print(f"   UMAP failed: {e}")
        return False
    
    # Test t-SNE
    print("\n2. Testing t-SNE...")
    try:
        tsne_result = reducer.fit_transform(X, method='tsne', n_components=2)
        print(f"   t-SNE successful! Shape: {tsne_result.shape}")
    except Exception as e:
        print(f"   t-SNE failed: {e}")
        return False
    
    # Test PCA for comparison
    print("\n3. Testing PCA...")
    try:
        pca_result = reducer.fit_transform(X, method='pca', n_components=2)
        print(f"   PCA successful! Shape: {pca_result.shape}")
    except Exception as e:
        print(f"   PCA failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_fixes()
    if not success:
        sys.exit(1)
