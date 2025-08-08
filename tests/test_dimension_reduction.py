"""Tests for dimension reduction algorithms."""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.dimension_reduction import DimensionReducer, PCA, TSNE, UMAP


class TestDimensionReducer:
    """Test cases for the main DimensionReducer class."""
    
    def setup_method(self) -> None:
        """Set up test data."""
        self.X = np.random.randn(100, 10)
        self.reducer = DimensionReducer()
    
    def test_available_methods(self) -> None:
        """Test that all expected methods are available."""
        methods = self.reducer.get_available_methods()
        expected_methods = ['pca', 'tsne', 'umap', 'autoencoder']
        
        for method in expected_methods:
            assert method in methods
    
    def test_pca_fit_transform(self) -> None:
        """Test PCA fit_transform method."""
        result = self.reducer.fit_transform(self.X, method='pca', n_components=2)
        
        assert result.shape == (100, 2)
        assert not np.any(np.isnan(result))
    
    def test_tsne_fit_transform(self) -> None:
        """Test t-SNE fit_transform method."""
        result = self.reducer.fit_transform(self.X, method='tsne', n_components=2)
        
        assert result.shape == (100, 2)
        assert not np.any(np.isnan(result))
    
    def test_umap_fit_transform(self) -> None:
        """Test UMAP fit_transform method."""
        result = self.reducer.fit_transform(self.X, method='umap', n_components=2)
        
        assert result.shape == (100, 2)
        assert not np.any(np.isnan(result))
    
    def test_invalid_method(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            self.reducer.fit_transform(self.X, method='invalid_method')
    
    def test_compare_methods(self) -> None:
        """Test compare_methods functionality."""
        methods = ['pca', 'umap']
        results = self.reducer.compare_methods(self.X, methods=methods, n_components=2)
        
        assert len(results) == 2
        assert 'PCA' in results
        assert 'UMAP' in results
        
        for method, result in results.items():
            assert result.shape == (100, 2)


class TestPCA:
    """Test cases for PCA implementation."""
    
    def setup_method(self) -> None:
        """Set up test data."""
        self.X = np.random.randn(100, 10)
        self.pca = PCA(n_components=2, random_state=42)
    
    def test_fit_transform(self) -> None:
        """Test PCA fit_transform."""
        result = self.pca.fit_transform(self.X)
        
        assert result.shape == (100, 2)
        assert not np.any(np.isnan(result))
        assert self.pca.fitted
    
    def test_inverse_transform(self) -> None:
        """Test PCA inverse_transform."""
        self.pca.fit(self.X)
        transformed = self.pca.transform(self.X)
        reconstructed = self.pca.inverse_transform(transformed)
        
        assert reconstructed.shape == self.X.shape
        assert not np.any(np.isnan(reconstructed))
    
    def test_explained_variance_ratio(self) -> None:
        """Test explained variance ratio property."""
        self.pca.fit(self.X)
        
        assert hasattr(self.pca, 'explained_variance_ratio_')
        assert len(self.pca.explained_variance_ratio_) == 2
        assert np.all(self.pca.explained_variance_ratio_ >= 0)
        assert np.sum(self.pca.explained_variance_ratio_) <= 1.0


class TestTSNE:
    """Test cases for t-SNE implementation."""
    
    def setup_method(self) -> None:
        """Set up test data."""
        self.X = np.random.randn(100, 10)
        self.tsne = TSNE(n_components=2, random_state=42, n_iter=100)
    
    def test_fit_transform(self) -> None:
        """Test t-SNE fit_transform."""
        result = self.tsne.fit_transform(self.X)
        
        assert result.shape == (100, 2)
        assert not np.any(np.isnan(result))
        assert self.tsne.fitted
    
    def test_no_transform_method(self) -> None:
        """Test that transform method raises NotImplementedError."""
        self.tsne.fit(self.X)
        
        with pytest.raises(NotImplementedError):
            self.tsne.transform(self.X)
    
    def test_embedding_property(self) -> None:
        """Test embedding property."""
        self.tsne.fit_transform(self.X)
        
        assert hasattr(self.tsne, 'embedding_')
        assert self.tsne.embedding_.shape == (100, 2)


class TestUMAP:
    """Test cases for UMAP implementation."""
    
    def setup_method(self) -> None:
        """Set up test data."""
        self.X = np.random.randn(100, 10)
        self.umap = UMAP(n_components=2, random_state=42)
    
    def test_fit_transform(self) -> None:
        """Test UMAP fit_transform."""
        result = self.umap.fit_transform(self.X)
        
        assert result.shape == (100, 2)
        assert not np.any(np.isnan(result))
        assert self.umap.fitted
    
    def test_fit_and_transform(self) -> None:
        """Test UMAP fit and transform separately."""
        self.umap.fit(self.X)
        result = self.umap.transform(self.X)
        
        assert result.shape == (100, 2)
        assert not np.any(np.isnan(result))
    
    def test_inverse_transform(self) -> None:
        """Test UMAP inverse_transform."""
        self.umap.fit(self.X)
        transformed = self.umap.transform(self.X)
        reconstructed = self.umap.inverse_transform(transformed)
        
        assert reconstructed.shape == self.X.shape
        assert not np.any(np.isnan(reconstructed))
    
    def test_embedding_property(self) -> None:
        """Test embedding property."""
        self.umap.fit(self.X)
        
        assert hasattr(self.umap, 'embedding_')
        assert self.umap.embedding_.shape == (100, 2)
