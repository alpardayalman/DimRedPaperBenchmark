"""Dimension reduction algorithms and utilities."""

from .base import BaseDimensionReducer
from .pca import PCA
from .tsne import TSNE
from .umap import UMAP
from .autoencoder import Autoencoder, AutoencoderReducer
from .isomap import ISOMAP
from .kernel_pca import KernelPCA
from .laplacian_eigenmap import LaplacianEigenmap
from .lle import LLE
from .vae import VAE, VAEReducer
from .main import DimensionReducer

__all__ = [
    "BaseDimensionReducer",
    "PCA",
    "TSNE", 
    "UMAP",
    "Autoencoder",
    "AutoencoderReducer",
    "ISOMAP",
    "KernelPCA",
    "LaplacianEigenmap",
    "LLE",
    "VAE",
    "VAEReducer",
    "DimensionReducer",
]
