"""Base metrics for benchmarking dimension reduction algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BenchmarkMetrics(ABC):
    """Abstract base class for benchmarking metrics.
    
    This class provides a common interface for all benchmarking metrics
    in the toolkit. It includes utility methods for common operations
    like distance calculations and nearest neighbor searches.
    
    Attributes:
        random_state: Random state for reproducibility.
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """Initialize the benchmark metrics.
        
        Args:
            random_state: Random state for reproducibility.
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def compute(self, X_original: "NDArray[np.floating]", X_reduced: "NDArray[np.floating]") -> float:
        """Compute the metric value.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            
        Returns:
            Metric value.
        """
        pass
    
    def compute_pairwise_distances(self, X: "NDArray[np.floating]", metric: str = 'euclidean') -> "NDArray[np.floating]":
        """Compute pairwise distances between all points.
        
        Args:
            X: Input data.
            metric: Distance metric to use.
            
        Returns:
            Pairwise distance matrix.
        """
        return pairwise_distances(X, metric=metric)
    
    def find_nearest_neighbors(
        self,
        X: "NDArray[np.floating]",
        n_neighbors: int,
        metric: str = 'euclidean'
    ) -> tuple["NDArray[np.floating]", "NDArray[np.floating]"]:
        """Find nearest neighbors for each point.
        
        Args:
            X: Input data.
            n_neighbors: Number of neighbors to find.
            metric: Distance metric to use.
            
        Returns:
            Tuple of (distances, indices) arrays.
        """
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)
        distances, indices = nbrs.kneighbors(X)
        return distances, indices
    
    def compute_neighborhood_preservation(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        n_neighbors: int = 10
    ) -> float:
        """Compute neighborhood preservation score.
        
        This measures how well the local neighborhood structure is preserved
        in the reduced space.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            n_neighbors: Number of neighbors to consider.
            
        Returns:
            Neighborhood preservation score (0-1, higher is better).
        """
        # Find neighbors in original space
        _, indices_original = self.find_nearest_neighbors(X_original, n_neighbors)
        
        # Find neighbors in reduced space
        _, indices_reduced = self.find_nearest_neighbors(X_reduced, n_neighbors)
        
        # Compute overlap
        overlaps = []
        for i in range(len(X_original)):
            original_neighbors = set(indices_original[i][1:])  # Exclude self
            reduced_neighbors = set(indices_reduced[i][1:])    # Exclude self
            overlap = len(original_neighbors.intersection(reduced_neighbors))
            overlaps.append(overlap / (n_neighbors - 1))
        
        return np.mean(overlaps)
    
    def compute_trustworthiness(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        n_neighbors: int = 10
    ) -> float:
        """Compute trustworthiness score.
        
        This measures how well the local structure is preserved in the
        reduced space by checking if points that are close in the reduced
        space were also close in the original space.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            n_neighbors: Number of neighbors to consider.
            
        Returns:
            Trustworthiness score (0-1, higher is better).
        """
        n_samples = len(X_original)
        
        # Find neighbors in reduced space
        _, indices_reduced = self.find_nearest_neighbors(X_reduced, n_neighbors)
        
        # Compute distances in original space
        distances_original = self.compute_pairwise_distances(X_original)
        
        # Compute trustworthiness
        trustworthiness = 0.0
        for i in range(n_samples):
            # Get neighbors in reduced space (excluding self)
            reduced_neighbors = indices_reduced[i][1:]
            
            # Count how many of these neighbors were actually close in original space
            rank_in_original = np.argsort(distances_original[i])
            close_in_original = set(rank_in_original[:n_neighbors])
            
            # Count violations
            violations = 0
            for j, neighbor in enumerate(reduced_neighbors):
                if neighbor not in close_in_original:
                    violations += j + 1
            
            # Normalize by maximum possible violations
            max_violations = (n_neighbors - 1) * n_neighbors // 2
            if max_violations > 0:
                trustworthiness += 1.0 - violations / max_violations
        
        return trustworthiness / n_samples
    
    def compute_continuity(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        n_neighbors: int = 10
    ) -> float:
        """Compute continuity score.
        
        This measures how well the global structure is preserved by checking
        if points that are close in the original space remain close in the
        reduced space.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            n_neighbors: Number of neighbors to consider.
            
        Returns:
            Continuity score (0-1, higher is better).
        """
        n_samples = len(X_original)
        
        # Find neighbors in original space
        _, indices_original = self.find_nearest_neighbors(X_original, n_neighbors)
        
        # Compute distances in reduced space
        distances_reduced = self.compute_pairwise_distances(X_reduced)
        
        # Compute continuity
        continuity = 0.0
        for i in range(n_samples):
            # Get neighbors in original space (excluding self)
            original_neighbors = indices_original[i][1:]
            
            # Count how many of these neighbors remain close in reduced space
            rank_in_reduced = np.argsort(distances_reduced[i])
            close_in_reduced = set(rank_in_reduced[:n_neighbors])
            
            # Count violations
            violations = 0
            for j, neighbor in enumerate(original_neighbors):
                if neighbor not in close_in_reduced:
                    violations += j + 1
            
            # Normalize by maximum possible violations
            max_violations = (n_neighbors - 1) * n_neighbors // 2
            if max_violations > 0:
                continuity += 1.0 - violations / max_violations
        
        return continuity / n_samples
    
    def compute_stress(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]"
    ) -> float:
        """Compute stress (Kruskal's stress).
        
        This measures the difference between distances in the original
        and reduced spaces.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            
        Returns:
            Stress value (lower is better).
        """
        # Compute pairwise distances
        distances_original = self.compute_pairwise_distances(X_original)
        distances_reduced = self.compute_pairwise_distances(X_reduced)
        
        # Compute stress
        numerator = np.sum((distances_original - distances_reduced) ** 2)
        denominator = np.sum(distances_original ** 2)
        
        if denominator == 0:
            return 0.0
        
        return np.sqrt(numerator / denominator)
    
    def compute_shepard_diagram_correlation(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]"
    ) -> float:
        """Compute correlation in Shepard diagram.
        
        This measures the correlation between distances in the original
        and reduced spaces.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            
        Returns:
            Correlation coefficient (-1 to 1, closer to 1 is better).
        """
        # Compute pairwise distances
        distances_original = self.compute_pairwise_distances(X_original)
        distances_reduced = self.compute_pairwise_distances(X_reduced)
        
        # Get upper triangular indices (avoid duplicates)
        upper_tri = np.triu_indices_from(distances_original, k=1)
        
        # Compute correlation
        corr = np.corrcoef(
            distances_original[upper_tri],
            distances_reduced[upper_tri]
        )[0, 1]
        
        return corr if not np.isnan(corr) else 0.0
