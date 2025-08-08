"""Uniform Manifold Approximation and Projection (UMAP) implementation."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import umap
from .base import BaseDimensionReducer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UMAP(BaseDimensionReducer):
    """Uniform Manifold Approximation and Projection (UMAP).
    
    UMAP is a manifold learning technique for dimension reduction that is
    particularly well-suited for embedding high-dimensional data in a
    low-dimensional space for visualization. It is based on Riemannian
    geometry and algebraic topology.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        n_neighbors: The size of local neighborhood for manifold approximation.
        min_dist: The effective minimum distance between embedded points.
        spread: The effective scale of embedded points.
        set_op_mix_ratio: Interpolate between fuzzy union and intersection.
        local_connectivity: The local connectivity required.
        repulsion_strength: Weighting applied to negative samples in low dimensional embedding.
        negative_sample_rate: The number of negative samples to select per positive sample.
        transform_queue_size: For transform operations, how many points to process at once.
        a: More specific parameters controlling the embedding.
        b: More specific parameters controlling the embedding.
        angular_rp_forest: Whether to use an angular random projection forest.
        target_n_neighbors: The number of neighbors to use in the target space.
        target_metric: The metric to use for the target space.
        target_metric_kwds: Additional arguments for the target metric.
        target_weight: Weighting factor between data topology and target topology.
        transform_seed: Random seed for transform operations.
        verbose: Whether to print status data during the computation.
        unique: If True, then the same input data will be mapped to the same output.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        set_op_mix_ratio: float = 1.0,
        local_connectivity: float = 1.0,
        repulsion_strength: float = 1.0,
        negative_sample_rate: int = 5,
        transform_queue_size: float = 4.0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        angular_rp_forest: bool = False,
        target_n_neighbors: int = -1,
        target_metric: str = "categorical",
        target_metric_kwds: Optional[dict] = None,
        target_weight: float = 0.5,
        transform_seed: int = 42,
        verbose: bool = False,
        unique: bool = False
    ) -> None:
        """Initialize UMAP.
        
        Args:
            n_components: Dimension of the embedded space.
            random_state: Random state for reproducibility.
            n_neighbors: The size of local neighborhood for manifold approximation.
            min_dist: The effective minimum distance between embedded points.
            spread: The effective scale of embedded points.
            set_op_mix_ratio: Interpolate between fuzzy union and intersection.
            local_connectivity: The local connectivity required.
            repulsion_strength: Weighting applied to negative samples in low dimensional embedding.
            negative_sample_rate: The number of negative samples to select per positive sample.
            transform_queue_size: For transform operations, how many points to process at once.
            a: More specific parameters controlling the embedding.
            b: More specific parameters controlling the embedding.
            angular_rp_forest: Whether to use an angular random projection forest.
            target_n_neighbors: The number of neighbors to use in the target space.
            target_metric: The metric to use for the target space.
            target_metric_kwds: Additional arguments for the target metric.
            target_weight: Weighting factor between data topology and target topology.
            transform_seed: Random seed for transform operations.
            verbose: Whether to print status data during the computation.
            unique: If True, then the same input data will be mapped to the same output.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.a = a
        self.b = b
        self.angular_rp_forest = angular_rp_forest
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.verbose = verbose
        self.unique = unique
        
        # Initialize UMAP
        self._umap = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            transform_queue_size=transform_queue_size,
            a=a,
            b=b,
            angular_rp_forest=angular_rp_forest,
            target_n_neighbors=target_n_neighbors,
            target_metric=target_metric,
            target_metric_kwds=target_metric_kwds,
            target_weight=target_weight,
            transform_seed=transform_seed,
            verbose=verbose,
            unique=unique
        )
    
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "UMAP":
        """Fit UMAP to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values for supervised dimension reduction.
            
        Returns:
            self: Fitted UMAP instance.
        """
        X = self._validate_data(X)
        self._umap.fit(X, y)
        self.fitted = True
        return self
    
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data using the fitted UMAP.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        return self._umap.transform(X)
    
    def fit_transform(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "NDArray[np.floating]":
        """Fit UMAP to the data and transform it.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values for supervised dimension reduction.
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        X = self._validate_data(X)
        result = self._umap.fit_transform(X, y)
        self.fitted = True
        return result
    
    def inverse_transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data back to original space.
        
        Args:
            X: Data in reduced space of shape (n_samples, n_components).
            
        Returns:
            Data in original space of shape (n_samples, n_features).
        """
        self._check_is_fitted()
        return self._umap.inverse_transform(X)
    
    @property
    def embedding_(self) -> "NDArray[np.floating]":
        """Stores the embedding vectors.
        
        Returns:
            Array of embedding vectors.
        """
        self._check_is_fitted()
        return self._umap.embedding_
    
    @property
    def graph_(self) -> "NDArray[np.floating]":
        """Stores the graph structure.
        
        Returns:
            Graph structure array.
        """
        self._check_is_fitted()
        return self._umap.graph_
    
    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit.
        
        Returns:
            Number of features.
        """
        self._check_is_fitted()
        return self._umap.n_features_in_
    
    @property
    def feature_names_in_(self) -> "NDArray[np.str_]":
        """Names of features seen during fit.
        
        Returns:
            Array of feature names.
        """
        self._check_is_fitted()
        return self._umap.feature_names_in_
    
    def get_embedding(self) -> "NDArray[np.floating]":
        """Get the final embedding.
        
        Returns:
            Final embedding array.
        """
        self._check_is_fitted()
        return self.embedding_
    
    def get_graph(self) -> "NDArray[np.floating]":
        """Get the graph structure.
        
        Returns:
            Graph structure array.
        """
        self._check_is_fitted()
        return self.graph_
