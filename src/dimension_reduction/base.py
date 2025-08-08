"""Base classes for dimension reduction algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BaseDimensionReducer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for dimension reduction algorithms.
    
    This class provides a common interface for all dimension reduction
    algorithms in the toolkit. It extends scikit-learn's BaseEstimator
    and TransformerMixin for compatibility with scikit-learn pipelines.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        fitted: Whether the model has been fitted.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the dimension reducer.
        
        Args:
            n_components: Number of components to reduce to.
            random_state: Random state for reproducibility.
            **kwargs: Additional algorithm-specific parameters.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.fitted = False
        self._set_params(**kwargs)
    
    def _set_params(self, **kwargs: Any) -> None:
        """Set algorithm-specific parameters.
        
        Args:
            **kwargs: Algorithm-specific parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "BaseDimensionReducer":
        """Fit the dimension reduction model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values (ignored for unsupervised methods).
            
        Returns:
            self: Fitted model instance.
        """
        pass
    
    @abstractmethod
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data to reduced dimensions.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        pass
    
    def fit_transform(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "NDArray[np.floating]":
        """Fit the model and transform the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values (ignored for unsupervised methods).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data back to original space.
        
        Args:
            X: Data in reduced space of shape (n_samples, n_components).
            
        Returns:
            Data in original space of shape (n_samples, n_features).
            
        Raises:
            NotImplementedError: If inverse transform is not supported.
        """
        raise NotImplementedError("Inverse transform not supported for this algorithm.")
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator and
                  contained subobjects that are estimators.
                  
        Returns:
            Parameter names mapped to their values.
        """
        params = {
            'n_components': self.n_components,
            'random_state': self.random_state,
        }
        
        if deep:
            # Add algorithm-specific parameters
            for key, value in self.__dict__.items():
                if not key.startswith('_') and key not in params:
                    params[key] = value
        
        return params
    
    def set_params(self, **params: Any) -> "BaseDimensionReducer":
        """Set the parameters of this estimator.
        
        Args:
            **params: Parameter names mapped to their values.
            
        Returns:
            self: Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self
    
    def _validate_data(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Validate input data.
        
        Args:
            X: Input data to validate.
            
        Returns:
            Validated data as numpy array.
            
        Raises:
            ValueError: If data is invalid.
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array")
        
        if X.shape[0] == 0:
            raise ValueError("Empty array")
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or infinite values")
        
        return X
    
    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted.
        
        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self.fitted:
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
