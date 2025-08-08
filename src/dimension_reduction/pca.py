"""Principal Component Analysis (PCA) implementation."""

from typing import Optional, TYPE_CHECKING
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from .base import BaseDimensionReducer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PCA(BaseDimensionReducer):
    """Principal Component Analysis for dimension reduction.
    
    PCA is a linear dimension reduction technique that finds the directions
    of maximum variance in the data and projects the data onto these
    principal components.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        explained_variance_ratio_: Percentage of variance explained by each component.
        singular_values_: Singular values corresponding to the components.
        components_: Principal axes in feature space.
        mean_: Per-feature empirical mean.
        n_components_: Number of components when n_components is 'mle'.
        noise_variance_: Estimated noise covariance following Tipping & Bishop 1999.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = 'auto',
        tol: float = 0.0,
        iterated_power: str = 'auto',
        n_oversamples: int = 10,
        power_iteration_normalizer: str = 'auto'
    ) -> None:
        """Initialize PCA.
        
        Args:
            n_components: Number of components to keep.
            random_state: Random state for reproducibility.
            copy: If False, data passed to fit are overwritten.
            whiten: When True, the components_ vectors are multiplied by the
                   square root of n_samples and then divided by the singular
                   values to ensure uncorrelated outputs with unit
                   component-wise variances.
            svd_solver: SVD solver to use: 'auto', 'full', 'arpack', 'randomized'.
            tol: Tolerance for singular values computed by svd_solver == 'arpack'.
            iterated_power: Number of iterations for the power method computed by
                           svd_solver == 'randomized'.
            n_oversamples: Additional number of random vectors to sample the range
                          of X so as to ensure proper conditioning.
            power_iteration_normalizer: Power iteration normalizer for randomized
                                       SVD solver.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        
        # Initialize sklearn PCA
        self._pca = SklearnPCA(
            n_components=n_components,
            random_state=random_state,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer
        )
    
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "PCA":
        """Fit PCA to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            self: Fitted PCA instance.
        """
        X = self._validate_data(X)
        self._pca.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data using the fitted PCA.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        return self._pca.transform(X)
    
    def inverse_transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data back to original space.
        
        Args:
            X: Data in reduced space of shape (n_samples, n_components).
            
        Returns:
            Data in original space of shape (n_samples, n_features).
        """
        self._check_is_fitted()
        return self._pca.inverse_transform(X)
    
    @property
    def explained_variance_ratio_(self) -> "NDArray[np.floating]":
        """Percentage of variance explained by each component.
        
        Returns:
            Array of explained variance ratios.
        """
        self._check_is_fitted()
        return self._pca.explained_variance_ratio_
    
    @property
    def singular_values_(self) -> "NDArray[np.floating]":
        """Singular values corresponding to the components.
        
        Returns:
            Array of singular values.
        """
        self._check_is_fitted()
        return self._pca.singular_values_
    
    @property
    def components_(self) -> "NDArray[np.floating]":
        """Principal axes in feature space.
        
        Returns:
            Array of principal components.
        """
        self._check_is_fitted()
        return self._pca.components_
    
    @property
    def mean_(self) -> "NDArray[np.floating]":
        """Per-feature empirical mean.
        
        Returns:
            Array of feature means.
        """
        self._check_is_fitted()
        return self._pca.mean_
    
    @property
    def n_components_(self) -> int:
        """Number of components when n_components is 'mle'.
        
        Returns:
            Number of components.
        """
        self._check_is_fitted()
        return self._pca.n_components_
    
    @property
    def noise_variance_(self) -> float:
        """Estimated noise covariance following Tipping & Bishop 1999.
        
        Returns:
            Estimated noise variance.
        """
        self._check_is_fitted()
        return self._pca.noise_variance_
    
    def get_cumulative_variance_ratio(self) -> "NDArray[np.floating]":
        """Get cumulative explained variance ratio.
        
        Returns:
            Cumulative explained variance ratio.
        """
        self._check_is_fitted()
        return np.cumsum(self.explained_variance_ratio_)
    
    def get_n_components_for_variance(self, variance_threshold: float) -> int:
        """Get number of components needed to explain a given variance threshold.
        
        Args:
            variance_threshold: Minimum variance to explain (0.0 to 1.0).
            
        Returns:
            Number of components needed.
        """
        self._check_is_fitted()
        cumulative_variance = self.get_cumulative_variance_ratio()
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        return min(n_components, len(cumulative_variance))
