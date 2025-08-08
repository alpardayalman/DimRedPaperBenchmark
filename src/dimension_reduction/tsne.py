"""t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation."""

from typing import Optional, TYPE_CHECKING
import numpy as np
from sklearn.manifold import TSNE as SklearnTSNE
from .base import BaseDimensionReducer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TSNE(BaseDimensionReducer):
    """t-Distributed Stochastic Neighbor Embedding (t-SNE).
    
    t-SNE is a non-linear dimension reduction technique that is particularly
    well-suited for embedding high-dimensional data in a low-dimensional
    space for visualization. It converts high-dimensional Euclidean
    distances between datapoints into conditional probabilities that
    represent similarities.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        perplexity: The perplexity is related to the number of nearest neighbors.
        early_exaggeration: Controls how tight natural clusters are in the embedding.
        learning_rate: The learning rate for t-SNE.
        n_iter: Maximum number of iterations for the optimization.
        n_iter_without_progress: Maximum number of iterations without progress.
        min_grad_norm: If the gradient norm is below this threshold, the optimization will be stopped.
        metric: The metric to use when calculating distance between instances.
        init: Initialization of embedding.
        verbose: Verbosity level.
        method: Gradient calculation method.
        angle: Only used if method='barnes_hut'.
        n_jobs: Number of jobs for parallel computation.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        n_iter_without_progress: int = 300,
        min_grad_norm: float = 1e-7,
        metric: str = "euclidean",
        init: str = "random",
        verbose: int = 0,
        method: str = "barnes_hut",
        angle: float = 0.5,
        n_jobs: Optional[int] = None
    ) -> None:
        """Initialize t-SNE.
        
        Args:
            n_components: Dimension of the embedded space.
            random_state: Random state for reproducibility.
            perplexity: The perplexity is related to the number of nearest neighbors.
            early_exaggeration: Controls how tight natural clusters are in the embedding.
            learning_rate: The learning rate for t-SNE.
            n_iter: Maximum number of iterations for the optimization.
            n_iter_without_progress: Maximum number of iterations without progress.
            min_grad_norm: If the gradient norm is below this threshold, the optimization will be stopped.
            metric: The metric to use when calculating distance between instances.
            init: Initialization of embedding.
            verbose: Verbosity level.
            method: Gradient calculation method.
            angle: Only used if method='barnes_hut'.
            n_jobs: Number of jobs for parallel computation.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        
        # Initialize sklearn t-SNE
        self._tsne = SklearnTSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            metric=metric,
            init=init,
            verbose=verbose,
            method=method,
            angle=angle,
            n_jobs=n_jobs
        )
    
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "TSNE":
        """Fit t-SNE to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            self: Fitted t-SNE instance.
        """
        X = self._validate_data(X)
        self._tsne.fit(X)
        self.fitted = True
        return self
    
    def fit_transform(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "NDArray[np.floating]":
        """Fit t-SNE to the data and transform it.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        X = self._validate_data(X)
        result = self._tsne.fit_transform(X)
        self.fitted = True
        return result
    
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data using the fitted t-SNE.
        
        Note: t-SNE does not support out-of-sample transformation.
        This method raises a NotImplementedError.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
            
        Raises:
            NotImplementedError: t-SNE does not support out-of-sample transformation.
        """
        raise NotImplementedError(
            "t-SNE does not support out-of-sample transformation. "
            "Use fit_transform() instead."
        )
    
    @property
    def embedding_(self) -> "NDArray[np.floating]":
        """Stores the embedding vectors.
        
        Returns:
            Array of embedding vectors.
        """
        self._check_is_fitted()
        return self._tsne.embedding_
    
    @property
    def kl_divergence_(self) -> float:
        """Kullback-Leibler divergence after optimization.
        
        Returns:
            KL divergence value.
        """
        self._check_is_fitted()
        return self._tsne.kl_divergence_
    
    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit.
        
        Returns:
            Number of features.
        """
        self._check_is_fitted()
        return self._tsne.n_features_in_
    
    @property
    def feature_names_in_(self) -> "NDArray[np.str_]":
        """Names of features seen during fit.
        
        Returns:
            Array of feature names.
        """
        self._check_is_fitted()
        return self._tsne.feature_names_in_
    
    def get_embedding(self) -> "NDArray[np.floating]":
        """Get the final embedding.
        
        Returns:
            Final embedding array.
        """
        self._check_is_fitted()
        return self.embedding_
    
    def get_kl_divergence(self) -> float:
        """Get the final KL divergence.
        
        Returns:
            Final KL divergence value.
        """
        self._check_is_fitted()
        return self.kl_divergence_
