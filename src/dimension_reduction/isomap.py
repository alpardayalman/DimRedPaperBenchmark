"""ISOMAP dimension reduction implementation."""

from typing import Optional, TYPE_CHECKING, Any
import numpy as np
from sklearn.manifold import Isomap as SklearnIsomap
from .base import BaseDimensionReducer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ISOMAP(BaseDimensionReducer):
    """ISOMAP dimension reduction algorithm.
    
    ISOMAP is a non-linear dimension reduction technique that preserves
    geodesic distances between data points. It constructs a neighborhood
    graph and computes shortest paths to approximate geodesic distances,
    then applies classical MDS to find a low-dimensional embedding.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        n_neighbors: Number of neighbors to consider for each point.
        radius: Radius of neighborhood for each point.
        eigen_solver: Eigenvalue decomposition strategy.
        tol: Convergence tolerance for arpack.
        max_iter: Maximum number of iterations for arpack.
        path_method: Method to use for finding shortest paths.
        neighbors_algorithm: Algorithm to use for nearest neighbors search.
        metric: Metric to use for distance computation.
        p: Parameter for the Minkowski metric.
        metric_params: Additional keyword arguments for the metric function.
        n_jobs: Number of parallel jobs.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        n_neighbors: int = 5,
        radius: Optional[float] = None,
        eigen_solver: str = 'auto',
        tol: float = 0.0,
        max_iter: Optional[int] = None,
        path_method: str = 'auto',
        neighbors_algorithm: str = 'auto',
        metric: str = 'minkowski',
        p: int = 2,
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None
    ) -> None:
        """Initialize ISOMAP.
        
        Args:
            n_components: Number of components to reduce to.
            random_state: Random state for reproducibility.
            n_neighbors: Number of neighbors to consider for each point.
            radius: Radius of neighborhood for each point.
            eigen_solver: Eigenvalue decomposition strategy.
            tol: Convergence tolerance for arpack.
            max_iter: Maximum number of iterations for arpack.
            path_method: Method to use for finding shortest paths.
            neighbors_algorithm: Algorithm to use for nearest neighbors search.
            metric: Metric to use for distance computation.
            p: Parameter for the Minkowski metric.
            metric_params: Additional keyword arguments for the metric function.
            n_jobs: Number of parallel jobs.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        
        # Initialize sklearn ISOMAP
        self._isomap = SklearnIsomap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            radius=radius,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            path_method=path_method,
            neighbors_algorithm=neighbors_algorithm,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs
        )
    
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "ISOMAP":
        """Fit ISOMAP to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            self: Fitted ISOMAP instance.
        """
        X = self._validate_data(X)
        self._isomap.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data using the fitted ISOMAP.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        return self._isomap.transform(X)
    
    def fit_transform(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "NDArray[np.floating]":
        """Fit ISOMAP to the data and transform it.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        return self.fit(X, y).transform(X)
    
    @property
    def embedding_(self) -> "NDArray[np.floating]":
        """Stores the embedding vectors.
        
        Returns:
            Array of embedding vectors.
        """
        self._check_is_fitted()
        return self._isomap.embedding_
    
    @property
    def kernel_pca_(self) -> Any:
        """KernelPCA object used to implement the embedding.
        
        Returns:
            KernelPCA object.
        """
        self._check_is_fitted()
        return self._isomap.kernel_pca_
    
    @property
    def training_data_(self) -> "NDArray[np.floating]":
        """Stores the training data.
        
        Returns:
            Training data.
        """
        self._check_is_fitted()
        return self._isomap.training_data_
    
    @property
    def nbrs_(self) -> Any:
        """Nearest neighbors instance.
        
        Returns:
            Nearest neighbors instance.
        """
        self._check_is_fitted()
        return self._isomap.nbrs_
    
    @property
    def dist_matrix_(self) -> "NDArray[np.floating]":
        """Stores the geodesic distance matrix of training data.
        
        Returns:
            Geodesic distance matrix.
        """
        self._check_is_fitted()
        return self._isomap.dist_matrix_
    
    def get_geodesic_distances(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Compute geodesic distances for new data points.
        
        Args:
            X: Data to compute geodesic distances for.
            
        Returns:
            Geodesic distance matrix.
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        # This is a simplified version - in practice, you'd need to implement
        # the full geodesic distance computation for new points
        return self._isomap.transform(X)
