"""Structure preservation metrics for benchmarking dimension reduction."""

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from .metrics import BenchmarkMetrics

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StructurePreservation(BenchmarkMetrics):
    """Structure preservation metrics for dimension reduction evaluation.
    
    This class provides comprehensive metrics for evaluating how well
    dimension reduction algorithms preserve the structure of the original
    data, including local and global structure preservation.
    
    Attributes:
        random_state: Random state for reproducibility.
        n_neighbors: Default number of neighbors for local metrics.
    """
    
    def __init__(self, random_state: Optional[int] = None, n_neighbors: int = 10) -> None:
        """Initialize structure preservation metrics.
        
        Args:
            random_state: Random state for reproducibility.
            n_neighbors: Default number of neighbors for local metrics.
        """
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
    
    def compute(self, X_original: "NDArray[np.floating]", X_reduced: "NDArray[np.floating]") -> Dict[str, float]:
        """Compute all structure preservation metrics.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            
        Returns:
            Dictionary of metric names and values.
        """
        return {
            'neighborhood_preservation': self.compute_neighborhood_preservation(X_original, X_reduced),
            'trustworthiness': self.compute_trustworthiness(X_original, X_reduced),
            'continuity': self.compute_continuity(X_original, X_reduced),
            'stress': self.compute_stress(X_original, X_reduced),
            'shepard_correlation': self.compute_shepard_diagram_correlation(X_original, X_reduced),
            'local_structure_score': self.compute_local_structure_score(X_original, X_reduced),
            'global_structure_score': self.compute_global_structure_score(X_original, X_reduced),
            'overall_structure_score': self.compute_overall_structure_score(X_original, X_reduced)
        }
    
    def compute_local_structure_score(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        n_neighbors: Optional[int] = None
    ) -> float:
        """Compute local structure preservation score.
        
        This combines neighborhood preservation and trustworthiness to
        provide a comprehensive local structure score.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            n_neighbors: Number of neighbors to consider.
            
        Returns:
            Local structure score (0-1, higher is better).
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        neighborhood_preservation = self.compute_neighborhood_preservation(
            X_original, X_reduced, n_neighbors
        )
        trustworthiness = self.compute_trustworthiness(
            X_original, X_reduced, n_neighbors
        )
        
        return (neighborhood_preservation + trustworthiness) / 2
    
    def compute_global_structure_score(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        n_neighbors: Optional[int] = None
    ) -> float:
        """Compute global structure preservation score.
        
        This combines continuity and Shepard correlation to provide
        a comprehensive global structure score.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            n_neighbors: Number of neighbors to consider.
            
        Returns:
            Global structure score (0-1, higher is better).
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        continuity = self.compute_continuity(X_original, X_reduced, n_neighbors)
        shepard_corr = self.compute_shepard_diagram_correlation(X_original, X_reduced)
        
        # Normalize Shepard correlation to 0-1 range
        shepard_score = (shepard_corr + 1) / 2
        
        return (continuity + shepard_score) / 2
    
    def compute_overall_structure_score(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        local_weight: float = 0.6,
        global_weight: float = 0.4
    ) -> float:
        """Compute overall structure preservation score.
        
        This combines local and global structure scores to provide
        a comprehensive evaluation of structure preservation.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            local_weight: Weight for local structure score.
            global_weight: Weight for global structure score.
            
        Returns:
            Overall structure score (0-1, higher is better).
        """
        local_score = self.compute_local_structure_score(X_original, X_reduced)
        global_score = self.compute_global_structure_score(X_original, X_reduced)
        
        return local_weight * local_score + global_weight * global_score
    
    def compute_multiscale_structure_analysis(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        neighbor_counts: List[int] = [5, 10, 20, 50]
    ) -> Dict[str, Dict[int, float]]:
        """Compute structure preservation at multiple scales.
        
        This analyzes how well structure is preserved at different
        neighborhood sizes, providing insights into the algorithm's
        behavior at different scales.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            neighbor_counts: List of neighbor counts to analyze.
            
        Returns:
            Dictionary mapping metric names to dictionaries of neighbor counts and scores.
        """
        results = {
            'neighborhood_preservation': {},
            'trustworthiness': {},
            'continuity': {},
            'local_structure_score': {}
        }
        
        for n_neighbors in neighbor_counts:
            results['neighborhood_preservation'][n_neighbors] = self.compute_neighborhood_preservation(
                X_original, X_reduced, n_neighbors
            )
            results['trustworthiness'][n_neighbors] = self.compute_trustworthiness(
                X_original, X_reduced, n_neighbors
            )
            results['continuity'][n_neighbors] = self.compute_continuity(
                X_original, X_reduced, n_neighbors
            )
            results['local_structure_score'][n_neighbors] = self.compute_local_structure_score(
                X_original, X_reduced, n_neighbors
            )
        
        return results
    
    def compute_structure_preservation_by_class(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        labels: "NDArray",
        n_neighbors: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute structure preservation metrics for each class.
        
        This analyzes how well structure is preserved within each class,
        which can reveal if certain classes are better preserved than others.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            labels: Class labels for each sample.
            n_neighbors: Number of neighbors to consider.
            
        Returns:
            Dictionary mapping class labels to dictionaries of metric names and values.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        unique_labels = np.unique(labels)
        results = {}
        
        for label in unique_labels:
            mask = labels == label
            X_orig_class = X_original[mask]
            X_reduced_class = X_reduced[mask]
            
            if len(X_orig_class) < n_neighbors + 1:
                # Skip classes with too few samples
                continue
            
            results[str(label)] = {
                'neighborhood_preservation': self.compute_neighborhood_preservation(
                    X_orig_class, X_reduced_class, n_neighbors
                ),
                'trustworthiness': self.compute_trustworthiness(
                    X_orig_class, X_reduced_class, n_neighbors
                ),
                'continuity': self.compute_continuity(
                    X_orig_class, X_reduced_class, n_neighbors
                ),
                'local_structure_score': self.compute_local_structure_score(
                    X_orig_class, X_reduced_class, n_neighbors
                )
            }
        
        return results
    
    def compute_distance_correlation_analysis(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        distance_thresholds: List[float] = None
    ) -> Dict[str, float]:
        """Compute distance correlation analysis.
        
        This analyzes how well distances are preserved at different
        distance thresholds, providing insights into local vs global
        distance preservation.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            distance_thresholds: List of distance thresholds to analyze.
            
        Returns:
            Dictionary of distance correlation metrics.
        """
        if distance_thresholds is None:
            # Use percentiles of original distances
            distances_original = self.compute_pairwise_distances(X_original)
            upper_tri = np.triu_indices_from(distances_original, k=1)
            original_distances = distances_original[upper_tri]
            distance_thresholds = [
                np.percentile(original_distances, p) for p in [25, 50, 75, 90]
            ]
        
        distances_original = self.compute_pairwise_distances(X_original)
        distances_reduced = self.compute_pairwise_distances(X_reduced)
        
        results = {}
        for threshold in distance_thresholds:
            # Find pairs within threshold in original space
            mask = distances_original <= threshold
            if np.sum(mask) > 0:
                corr = np.corrcoef(
                    distances_original[mask],
                    distances_reduced[mask]
                )[0, 1]
                results[f'distance_corr_threshold_{threshold:.3f}'] = corr if not np.isnan(corr) else 0.0
        
        return results
