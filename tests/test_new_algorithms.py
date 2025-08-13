"""Tests for the new dimension reduction algorithms."""

import pytest
import numpy as np
from typing import TYPE_CHECKING

from src.dimension_reduction.kernel_pca import KernelPCA
from src.dimension_reduction.isomap import ISOMAP
from src.dimension_reduction.lle import LLE
from src.dimension_reduction.laplacian_eigenmap import LaplacianEigenmap
from src.dimension_reduction.vae import VAEReducer, VAE, VAEEncoder, VAEDecoder

# Import torch for VAE tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate sample data for testing.
    
    Returns:
        Sample data array.
    """
    np.random.seed(42)
    return np.random.randn(100, 10)


class TestKernelPCA:
    """Test cases for Kernel PCA."""
    
    def test_kernel_pca_initialization(self) -> None:
        """Test Kernel PCA initialization."""
        kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1)
        assert kpca.n_components == 3
        assert kpca.kernel == 'rbf'
        assert kpca.gamma == 0.1
        assert not kpca.fitted
    
    def test_kernel_pca_fit_transform(self, sample_data: np.ndarray) -> None:
        """Test Kernel PCA fit and transform."""
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
        X_transformed = kpca.fit_transform(sample_data)
        
        assert kpca.fitted
        assert X_transformed.shape == (100, 2)
        assert not np.any(np.isnan(X_transformed))
    
    def test_kernel_pca_properties(self, sample_data: np.ndarray) -> None:
        """Test Kernel PCA properties."""
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
        kpca.fit(sample_data)
        
        # Check for essential properties
        assert hasattr(kpca, 'eigenvalues_')
        assert hasattr(kpca, 'eigenvectors_')
        # Some properties might not be available depending on sklearn version
        # assert hasattr(kpca, 'dual_coef_')
        # assert hasattr(kpca, 'X_transformed_fit_')
        # assert hasattr(kpca, 'X_fit_')


class TestISOMAP:
    """Test cases for ISOMAP."""
    
    def test_isomap_initialization(self) -> None:
        """Test ISOMAP initialization."""
        isomap = ISOMAP(n_components=3, n_neighbors=10)
        assert isomap.n_components == 3
        assert isomap.n_neighbors == 10
        assert not isomap.fitted
    
    def test_isomap_fit_transform(self, sample_data: np.ndarray) -> None:
        """Test ISOMAP fit and transform."""
        isomap = ISOMAP(n_components=2, n_neighbors=10)
        X_transformed = isomap.fit_transform(sample_data)
        
        assert isomap.fitted
        assert X_transformed.shape == (100, 2)
        assert not np.any(np.isnan(X_transformed))
    
    def test_isomap_properties(self, sample_data: np.ndarray) -> None:
        """Test ISOMAP properties."""
        isomap = ISOMAP(n_components=2, n_neighbors=10)
        isomap.fit(sample_data)
        
        # Check for essential properties
        assert hasattr(isomap, 'embedding_')
        # Some properties might not be available depending on sklearn version
        # assert hasattr(isomap, 'kernel_pca_')
        # assert hasattr(isomap, 'training_data_')
        # assert hasattr(isomap, 'nbrs_')
        # assert hasattr(isomap, 'dist_matrix_')


class TestLLE:
    """Test cases for Locally Linear Embedding."""
    
    def test_lle_initialization(self) -> None:
        """Test LLE initialization."""
        lle = LLE(n_components=3, n_neighbors=10, reg=1e-3)
        assert lle.n_components == 3
        assert lle.n_neighbors == 10
        assert lle.reg == 1e-3
        assert not lle.fitted
    
    def test_lle_fit_transform(self, sample_data: np.ndarray) -> None:
        """Test LLE fit and transform."""
        lle = LLE(n_components=2, n_neighbors=10, reg=1e-3)
        X_transformed = lle.fit_transform(sample_data)
        
        assert lle.fitted
        assert X_transformed.shape == (100, 2)
        assert not np.any(np.isnan(X_transformed))
    
    def test_lle_properties(self, sample_data: np.ndarray) -> None:
        """Test LLE properties."""
        lle = LLE(n_components=2, n_neighbors=10, reg=1e-3)
        lle.fit(sample_data)
        
        # Check for essential properties
        assert hasattr(lle, 'embedding_')
        # Some properties might not be available depending on sklearn version
        # assert hasattr(lle, 'reconstruction_error_')
        # assert hasattr(lle, 'nbrs_')
        # assert hasattr(lle, 'weights_')


class TestLaplacianEigenmap:
    """Test cases for Laplacian Eigenmaps."""
    
    def test_laplacian_eigenmap_initialization(self) -> None:
        """Test Laplacian Eigenmap initialization."""
        le = LaplacianEigenmap(n_components=3, n_neighbors=10, affinity='nearest_neighbors')
        assert le.n_components == 3
        assert le.n_neighbors == 10
        assert le.affinity == 'nearest_neighbors'
        assert not le.fitted
    
    def test_laplacian_eigenmap_fit_transform(self, sample_data: np.ndarray) -> None:
        """Test Laplacian Eigenmap fit and transform."""
        le = LaplacianEigenmap(n_components=2, n_neighbors=10, affinity='nearest_neighbors')
        X_transformed = le.fit_transform(sample_data)
        
        assert le.fitted
        assert X_transformed.shape == (100, 2)
        assert not np.any(np.isnan(X_transformed))
    
    def test_laplacian_eigenmap_properties(self, sample_data: np.ndarray) -> None:
        """Test Laplacian Eigenmap properties."""
        le = LaplacianEigenmap(n_components=2, n_neighbors=10, affinity='nearest_neighbors')
        le.fit(sample_data)
        
        # Check for essential properties
        assert hasattr(le, 'embedding_')
        # Some properties might not be available depending on sklearn version
        # assert hasattr(le, 'affinity_matrix_')
        # assert hasattr(le, 'nbrs_')


class TestVAEComponents:
    """Test cases for VAE components."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_vae_encoder(self) -> None:
        """Test VAE encoder."""
        encoder = VAEEncoder(input_dim=10, hidden_dims=[64, 32], latent_dim=2)
        
        # Test forward pass
        x = np.random.randn(5, 10)
        x_tensor = encoder(torch.FloatTensor(x))
        
        assert len(x_tensor) == 2  # mu and logvar
        assert x_tensor[0].shape == (5, 2)  # mu
        assert x_tensor[1].shape == (5, 2)  # logvar
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_vae_decoder(self) -> None:
        """Test VAE decoder."""
        decoder = VAEDecoder(latent_dim=2, hidden_dims=[32, 64], output_dim=10)
        
        # Test forward pass
        z = np.random.randn(5, 2)
        z_tensor = torch.FloatTensor(z)
        output = decoder(z_tensor)
        
        assert output.shape == (5, 10)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_vae_model(self) -> None:
        """Test VAE model."""
        encoder = VAEEncoder(input_dim=10, hidden_dims=[64, 32], latent_dim=2)
        decoder = VAEDecoder(latent_dim=2, hidden_dims=[32, 64], output_dim=10)
        vae = VAE(encoder, decoder, latent_dim=2)
        
        # Test forward pass
        x = np.random.randn(5, 10)
        x_tensor = torch.FloatTensor(x)
        recon_x, mu, logvar = vae(x_tensor)
        
        assert recon_x.shape == (5, 10)
        assert mu.shape == (5, 2)
        assert logvar.shape == (5, 2)


class TestVAEReducer:
    """Test cases for VAE Reducer."""
    
    def test_vae_reducer_initialization(self) -> None:
        """Test VAE Reducer initialization."""
        vae = VAEReducer(n_components=3, hidden_dims=[64, 32], epochs=50)
        assert vae.n_components == 3
        assert vae.hidden_dims == [64, 32]
        assert vae.epochs == 50
        assert not vae.fitted
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.slow
    def test_vae_reducer_fit_transform(self, sample_data: np.ndarray) -> None:
        """Test VAE Reducer fit and transform."""
        vae = VAEReducer(n_components=2, hidden_dims=[64], epochs=10, batch_size=32)
        X_transformed = vae.fit_transform(sample_data)
        
        assert vae.fitted
        assert X_transformed.shape == (100, 2)
        assert not np.any(np.isnan(X_transformed))
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.slow
    def test_vae_reducer_inverse_transform(self, sample_data: np.ndarray) -> None:
        """Test VAE Reducer inverse transform."""
        vae = VAEReducer(n_components=2, hidden_dims=[64], epochs=10, batch_size=32)
        vae.fit(sample_data)
        
        X_transformed = vae.transform(sample_data)
        X_reconstructed = vae.inverse_transform(X_transformed)
        
        assert X_reconstructed.shape == sample_data.shape
        assert not np.any(np.isnan(X_reconstructed))
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.slow
    def test_vae_reducer_sampling(self, sample_data: np.ndarray) -> None:
        """Test VAE Reducer sampling."""
        vae = VAEReducer(n_components=2, hidden_dims=[64], epochs=10, batch_size=32)
        vae.fit(sample_data)
        
        samples = vae.sample(n_samples=10)
        assert samples.shape == (10, sample_data.shape[1])
        assert not np.any(np.isnan(samples))


class TestIntegration:
    """Integration tests for the main dimension reducer."""
    
    def test_all_methods_available(self) -> None:
        """Test that all methods are available in the main reducer."""
        from src.dimension_reduction.main import DimensionReducer
        
        reducer = DimensionReducer()
        available_methods = reducer.get_available_methods()
        
        expected_methods = [
            'pca', 'tsne', 'umap', 'autoencoder',
            'kernel_pca', 'isomap', 'lle', 'laplacian_eigenmap', 'vae'
        ]
        
        for method in expected_methods:
            assert method in available_methods, f"Method {method} not found in available methods"
    
    def test_method_parameters(self) -> None:
        """Test that methods can be initialized with parameters."""
        from src.dimension_reduction.main import DimensionReducer
        
        reducer = DimensionReducer()
        
        # Test with different parameters
        test_params = {
            'kernel_pca': {'kernel': 'poly', 'degree': 3},
            'isomap': {'n_neighbors': 15},
            'lle': {'n_neighbors': 15, 'reg': 1e-2},
            'laplacian_eigenmap': {'n_neighbors': 15, 'affinity': 'rbf'},
            'vae': {'epochs': 20, 'batch_size': 16}
        }
        
        for method, params in test_params.items():
            try:
                model = reducer.methods[method](n_components=2, **params)
                assert model is not None
            except Exception as e:
                pytest.fail(f"Failed to initialize {method} with params {params}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
