"""Main dimension reduction interface."""

from typing import Any, Dict, Optional, TYPE_CHECKING
import numpy as np
from .base import BaseDimensionReducer
from .pca import PCA
from .tsne import TSNE
from .umap import UMAP
from .autoencoder import AutoencoderReducer
from .kernel_pca import KernelPCA
from .lle import LLE
from .vae import VAEReducer
from .isomap import ISOMAP
from .laplacian_eigenmap import LaplacianEigenmap

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DimensionReducer:
    """Unified interface for dimension reduction algorithms.
    
    This class provides a simple interface to access all dimension reduction
    methods in the toolkit. It automatically handles method selection and
    parameter configuration.
    
    Attributes:
        methods: Dictionary mapping method names to their implementations.
    """
    
    def __init__(self) -> None:
        """Initialize the dimension reducer."""
        self.methods = {
            'pca': PCA,
            'tsne': TSNE,
            'umap': UMAP,
            'autoencoder': AutoencoderReducer,
            'kernel_pca': KernelPCA,
            'lle': LLE,
            'vae': VAEReducer,
            'isomap': ISOMAP,
            'laplacian_eigenmap': LaplacianEigenmap
        }
        self._fitted_models: Dict[str, BaseDimensionReducer] = {}
    
    def fit_transform(
        self,
        X: "NDArray[np.floating]",
        method: str = 'umap',
        n_components: int = 2,
        **kwargs: Any
    ) -> "NDArray[np.floating]":
        """Fit and transform data using the specified method.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            method: Dimension reduction method ('pca', 'tsne', 'umap', 'autoencoder').
            n_components: Number of components to reduce to.
            **kwargs: Additional method-specific parameters.
            
        Returns:
            Transformed data of shape (n_samples, n_components).
            
        Raises:
            ValueError: If method is not supported.
        """
        if method not in self.methods:
            raise ValueError(f"Method '{method}' not supported. Available methods: {list(self.methods.keys())}")
        
        # Create and fit model
        model_class = self.methods[method]
        model = model_class(n_components=n_components, **kwargs)
        
        # Store fitted model
        self._fitted_models[method] = model
        
        # Fit and transform
        return model.fit_transform(X)
    
    def fit(
        self,
        X: "NDArray[np.floating]",
        method: str = 'umap',
        n_components: int = 2,
        **kwargs: Any
    ) -> BaseDimensionReducer:
        """Fit a dimension reduction model.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            method: Dimension reduction method ('pca', 'tsne', 'umap', 'autoencoder').
            n_components: Number of components to reduce to.
            **kwargs: Additional method-specific parameters.
            
        Returns:
            Fitted model instance.
            
        Raises:
            ValueError: If method is not supported.
        """
        if method not in self.methods:
            raise ValueError(f"Method '{method}' not supported. Available methods: {list(self.methods.keys())}")
        
        # Create and fit model
        model_class = self.methods[method]
        model = model_class(n_components=n_components, **kwargs)
        
        # Store fitted model
        self._fitted_models[method] = model
        
        # Fit
        return model.fit(X)
    
    def transform(
        self,
        X: "NDArray[np.floating]",
        method: str = 'umap'
    ) -> "NDArray[np.floating]":
        """Transform data using a fitted model.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            method: Dimension reduction method to use.
            
        Returns:
            Transformed data.
            
        Raises:
            ValueError: If method is not fitted or not supported.
        """
        if method not in self._fitted_models:
            raise ValueError(f"Method '{method}' not fitted. Call fit() first.")
        
        return self._fitted_models[method].transform(X)
    
    def get_model(self, method: str) -> Optional[BaseDimensionReducer]:
        """Get a fitted model.
        
        Args:
            method: Dimension reduction method.
            
        Returns:
            Fitted model or None if not fitted.
        """
        return self._fitted_models.get(method)
    
    def get_available_methods(self) -> list[str]:
        """Get list of available methods.
        
        Returns:
            List of available method names.
        """
        return list(self.methods.keys())
    
    def get_fitted_methods(self) -> list[str]:
        """Get list of fitted methods.
        
        Returns:
            List of fitted method names.
        """
        return list(self._fitted_models.keys())
    
    def compare_methods(
        self,
        X: "NDArray[np.floating]",
        methods: Optional[list[str]] = None,
        n_components: int = 2,
        method_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, "NDArray[np.floating]"]:
        """Compare multiple dimension reduction methods.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            methods: List of methods to compare. If None, uses all available methods.
            n_components: Number of components to reduce to.
            method_params: Dictionary mapping method names to their parameters.
            
        Returns:
            Dictionary mapping method names to their transformed data.
        """
        if methods is None:
            methods = self.get_available_methods()
        
        if method_params is None:
            method_params = {}
        
        results = {}
        for method in methods:
            if method not in self.methods:
                print(f"Warning: Method '{method}' not supported, skipping.")
                continue
            
            try:
                params = method_params.get(method, {})
                result = self.fit_transform(X, method, n_components, **params)
                results[method] = result
                print(f"Successfully applied {method}")
            except Exception as e:
                print(f"Error applying {method}: {e}")
        
        return results
    
    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get information about a method.
        
        Args:
            method: Dimension reduction method.
            
        Returns:
            Dictionary with method information.
            
        Raises:
            ValueError: If method is not supported.
        """
        if method not in self.methods:
            raise ValueError(f"Method '{method}' not supported. Available methods: {list(self.methods.keys())}")
        
        model_class = self.methods[method]
        model = model_class()
        
        return {
            'name': method,
            'class': model_class,
            'docstring': model_class.__doc__,
            'parameters': model.get_params(),
            'supports_inverse_transform': hasattr(model, 'inverse_transform')
        }
