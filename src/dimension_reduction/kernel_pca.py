"""Kernel Principal Component Analysis (Kernel PCA) implementation."""

from typing import Optional, TYPE_CHECKING
import numpy as np
from sklearn.decomposition import KernelPCA as SklearnKernelPCA
from .base import BaseDimensionReducer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class KernelPCA(BaseDimensionReducer):
    """Kernel Principal Component Analysis for dimension reduction.
    
    Kernel PCA is a non-linear dimension reduction technique that applies
    PCA in a higher-dimensional feature space defined by a kernel function.
    This allows it to capture non-linear relationships in the data.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        kernel: Kernel function to use ('rbf', 'poly', 'linear', 'cosine', 'sigmoid').
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
        degree: Degree of polynomial kernel function.
        coef0: Independent term in polynomial and sigmoid kernels.
        alpha: Regularization parameter for ridge regression.
        fit_inverse_transform: Whether to fit inverse transform.
        eigen_solver: Eigenvalue decomposition strategy.
        tol: Convergence tolerance for arpack.
        max_iter: Maximum number of iterations for arpack.
        remove_zero_eig: If True, remove components with zero eigenvalues.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        kernel: str = 'rbf',
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        alpha: float = 1.0,
        fit_inverse_transform: bool = False,
        eigen_solver: str = 'auto',
        tol: float = 0.0,
        max_iter: Optional[int] = None,
        remove_zero_eig: bool = False
    ) -> None:
        """Initialize Kernel PCA.
        
        Args:
            n_components: Number of components to keep.
            random_state: Random state for reproducibility.
            kernel: Kernel function to use.
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
            degree: Degree of polynomial kernel function.
            coef0: Independent term in polynomial and sigmoid kernels.
            alpha: Regularization parameter for ridge regression.
            fit_inverse_transform: Whether to fit inverse transform.
            eigen_solver: Eigenvalue decomposition strategy.
            tol: Convergence tolerance for arpack.
            max_iter: Maximum number of iterations for arpack.
            remove_zero_eig: If True, remove components with zero eigenvalues.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        
        # Initialize sklearn Kernel PCA
        self._kernel_pca = SklearnKernelPCA(
            n_components=n_components,
            random_state=random_state,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            remove_zero_eig=remove_zero_eig
        )
    
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "KernelPCA":
        """Fit Kernel PCA to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            self: Fitted Kernel PCA instance.
        """
        X = self._validate_data(X)
        self._kernel_pca.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data using the fitted Kernel PCA.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        return self._kernel_pca.transform(X)
    
    def inverse_transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data back to original space.
        
        Args:
            X: Data in reduced space of shape (n_samples, n_components).
            
        Returns:
            Data in original space of shape (n_samples, n_features).
            
        Raises:
            ValueError: If inverse transform is not fitted.
        """
        self._check_is_fitted()
        if not self.fit_inverse_transform:
            raise ValueError("Inverse transform not fitted. Set fit_inverse_transform=True during initialization.")
        return self._kernel_pca.inverse_transform(X)
    
    @property
    def eigenvalues_(self) -> "NDArray[np.floating]":
        """Eigenvalues of the centered kernel matrix.
        
        Returns:
            Array of eigenvalues.
        """
        self._check_is_fitted()
        return self._kernel_pca.eigenvalues_
    
    @property
    def eigenvectors_(self) -> "NDArray[np.floating]":
        """Eigenvectors of the centered kernel matrix.
        
        Returns:
            Array of eigenvectors.
        """
        self._check_is_fitted()
        return self._kernel_pca.eigenvectors_
    
    @property
    def dual_coef_(self) -> "NDArray[np.floating]":
        """Inverse transform matrix.
        
        Returns:
            Inverse transform matrix.
        """
        self._check_is_fitted()
        return self._kernel_pca.dual_coef_
    
    @property
    def X_transformed_fit_(self) -> "NDArray[np.floating]":
        """Projection of the fitted data.
        
        Returns:
            Projection of the fitted data.
        """
        self._check_is_fitted()
        return self._kernel_pca.X_transformed_fit_
    
    @property
    def X_fit_(self) -> "NDArray[np.floating]":
        """Training data.
        
        Returns:
            Training data.
        """
        self._check_is_fitted()
        return self._kernel_pca.X_fit_
    
    def get_kernel_matrix(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Compute the kernel matrix for given data.
        
        Args:
            X: Data to compute kernel matrix for.
            
        Returns:
            Kernel matrix.
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        return self._kernel_pca._get_kernel(X, self.X_fit_)
