"""Visualization quality metrics for benchmarking dimension reduction."""

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from .metrics import BenchmarkMetrics

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VisualizationQuality(BenchmarkMetrics):
    """Visualization quality metrics for dimension reduction evaluation.
    
    This class provides metrics for evaluating the quality of visualizations
    produced by dimension reduction algorithms, focusing on class separation,
    compactness, and overall visual appeal.
    
    Attributes:
        random_state: Random state for reproducibility.
        n_neighbors: Default number of neighbors for local metrics.
    """
    
    def __init__(self, random_state: Optional[int] = None, n_neighbors: int = 10) -> None:
        """Initialize visualization quality metrics.
        
        Args:
            random_state: Random state for reproducibility.
            n_neighbors: Default number of neighbors for local metrics.
        """
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
    
    def compute(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> Dict[str, float]:
        """Compute all visualization quality metrics.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels for supervised evaluation.
            
        Returns:
            Dictionary of metric names and values.
        """
        metrics = {
            'point_density_score': self.compute_point_density_score(X),
            'spatial_distribution_score': self.compute_spatial_distribution_score(X),
            'overall_visualization_score': self.compute_overall_visualization_score(X)
        }
        
        if y is not None:
            metrics.update({
                'class_separation': self.compute_class_separation(X, y),
                'class_compactness': self.compute_class_compactness(X, y),
                'inter_class_distance': self.compute_inter_class_distance(X, y),
                'intra_class_distance': self.compute_intra_class_distance(X, y),
                'class_overlap': self.compute_class_overlap(X, y)
            })
        
        return metrics
    
    def compute_class_separation(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> float:
        """Compute class separation score.
        
        This measures how well-separated different classes are in the
        visualization space.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels.
            
        Returns:
            Class separation score (0-1, higher is better).
        """
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return 0.0
        
        # Compute class centroids
        centroids = []
        for label in unique_labels:
            class_points = X[y == label]
            centroid = np.mean(class_points, axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Compute average distance between class centroids
        centroid_distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                centroid_distances.append(distance)
        
        if not centroid_distances:
            return 0.0
        
        # Normalize by data range
        data_range = np.max(X) - np.min(X)
        if data_range == 0:
            return 0.0
        
        return min(np.mean(centroid_distances) / data_range, 1.0)
    
    def compute_class_compactness(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> float:
        """Compute class compactness score.
        
        This measures how compact each class is within itself.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels.
            
        Returns:
            Class compactness score (0-1, higher is better).
        """
        unique_labels = np.unique(y)
        if len(unique_labels) < 1:
            return 0.0
        
        compactness_scores = []
        data_range = np.max(X) - np.min(X)
        
        for label in unique_labels:
            class_points = X[y == label]
            if len(class_points) < 2:
                continue
            
            # Compute centroid of the class
            centroid = np.mean(class_points, axis=0)
            
            # Compute average distance from centroid
            distances = np.linalg.norm(class_points - centroid, axis=1)
            avg_distance = np.mean(distances)
            
            # Convert to compactness score (lower distances = higher compactness)
            if data_range > 0:
                compactness = max(0, 1 - avg_distance / data_range)
                compactness_scores.append(compactness)
        
        if not compactness_scores:
            return 0.0
        
        return np.mean(compactness_scores)
    
    def compute_inter_class_distance(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> float:
        """Compute average inter-class distance.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels.
            
        Returns:
            Average inter-class distance.
        """
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return 0.0
        
        inter_class_distances = []
        
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i+1:], i+1):
                class1_points = X[y == label1]
                class2_points = X[y == label2]
                
                # Compute pairwise distances between classes
                distances = pairwise_distances(class1_points, class2_points)
                inter_class_distances.extend(distances.flatten())
        
        if not inter_class_distances:
            return 0.0
        
        return np.mean(inter_class_distances)
    
    def compute_intra_class_distance(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> float:
        """Compute average intra-class distance.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels.
            
        Returns:
            Average intra-class distance.
        """
        unique_labels = np.unique(y)
        if len(unique_labels) < 1:
            return 0.0
        
        intra_class_distances = []
        
        for label in unique_labels:
            class_points = X[y == label]
            if len(class_points) < 2:
                continue
            
            # Compute pairwise distances within class
            distances = pairwise_distances(class_points)
            # Get upper triangular part (excluding diagonal)
            upper_tri = np.triu_indices_from(distances, k=1)
            intra_class_distances.extend(distances[upper_tri])
        
        if not intra_class_distances:
            return 0.0
        
        return np.mean(intra_class_distances)
    
    def compute_class_overlap(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray",
        n_neighbors: Optional[int] = None
    ) -> float:
        """Compute class overlap score.
        
        This measures the degree of overlap between different classes
        in the visualization space.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels.
            n_neighbors: Number of neighbors to consider.
            
        Returns:
            Class overlap score (0-1, lower is better).
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return 0.0
        
        overlap_scores = []
        
        for i in range(len(X)):
            # Find neighbors of point i
            distances = np.linalg.norm(X - X[i], axis=1)
            neighbor_indices = np.argsort(distances)[1:n_neighbors+1]  # Exclude self
            
            # Count neighbors from different classes
            neighbor_labels = y[neighbor_indices]
            current_label = y[i]
            different_class_neighbors = np.sum(neighbor_labels != current_label)
            
            overlap_score = different_class_neighbors / n_neighbors
            overlap_scores.append(overlap_score)
        
        return np.mean(overlap_scores)
    
    def compute_point_density_score(
        self,
        X: "NDArray[np.floating]"
    ) -> float:
        """Compute point density score.
        
        This measures how well the points are distributed in the
        visualization space without excessive clustering or sparsity.
        
        Args:
            X: Reduced data for visualization.
            
        Returns:
            Point density score (0-1, higher is better).
        """
        # Compute pairwise distances
        distances = self.compute_pairwise_distances(X)
        
        # Get upper triangular part (excluding diagonal)
        upper_tri = np.triu_indices_from(distances, k=1)
        all_distances = distances[upper_tri]
        
        if len(all_distances) == 0:
            return 0.0
        
        # Compute distance statistics
        mean_distance = np.mean(all_distances)
        std_distance = np.std(all_distances)
        
        # Compute coefficient of variation
        cv = std_distance / mean_distance if mean_distance > 0 else 0
        
        # Convert to density score (lower CV = more uniform distribution)
        density_score = max(0, 1 - cv)
        
        return density_score
    
    def compute_spatial_distribution_score(
        self,
        X: "NDArray[np.floating]"
    ) -> float:
        """Compute spatial distribution score.
        
        This measures how well the points are distributed across
        the visualization space.
        
        Args:
            X: Reduced data for visualization.
            
        Returns:
            Spatial distribution score (0-1, higher is better).
        """
        # Compute bounding box
        min_coords = np.min(X, axis=0)
        max_coords = np.max(X, axis=0)
        range_coords = max_coords - min_coords
        
        if np.any(range_coords == 0):
            return 0.0
        
        # Normalize coordinates to [0, 1]
        X_normalized = (X - min_coords) / range_coords
        
        # Divide space into grid cells
        n_cells = 10  # 10x10 grid for 2D
        cell_size = 1.0 / n_cells
        
        # Count points in each cell
        cell_counts = np.zeros((n_cells, n_cells))
        for point in X_normalized:
            cell_x = min(int(point[0] / cell_size), n_cells - 1)
            cell_y = min(int(point[1] / cell_size), n_cells - 1)
            cell_counts[cell_x, cell_y] += 1
        
        # Compute distribution uniformity
        expected_count = len(X) / (n_cells * n_cells)
        if expected_count == 0:
            return 0.0
        
        # Compute coefficient of variation of cell counts
        cv = np.std(cell_counts) / np.mean(cell_counts) if np.mean(cell_counts) > 0 else 0
        
        # Convert to distribution score (lower CV = more uniform)
        distribution_score = max(0, 1 - cv)
        
        return distribution_score
    
    def compute_overall_visualization_score(
        self,
        X: "NDArray[np.floating]",
        y: Optional["NDArray"] = None
    ) -> float:
        """Compute overall visualization quality score.
        
        This combines multiple visualization metrics to provide a
        comprehensive assessment of visualization quality.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels for supervised evaluation.
            
        Returns:
            Overall visualization quality score (0-1, higher is better).
        """
        scores = []
        
        # Unsupervised metrics
        scores.append(self.compute_point_density_score(X))
        scores.append(self.compute_spatial_distribution_score(X))
        
        # Supervised metrics (if labels available)
        if y is not None:
            scores.append(self.compute_class_separation(X, y))
            scores.append(self.compute_class_compactness(X, y))
            # Invert overlap score (lower is better)
            overlap_score = self.compute_class_overlap(X, y)
            scores.append(1 - overlap_score)
        
        return np.mean(scores)
    
    def compute_visualization_by_class(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> Dict[str, Dict[str, float]]:
        """Compute visualization metrics for each class.
        
        Args:
            X: Reduced data for visualization.
            y: Target labels.
            
        Returns:
            Dictionary mapping class labels to their visualization metrics.
        """
        unique_labels = np.unique(y)
        results = {}
        
        for label in unique_labels:
            mask = y == label
            X_class = X[mask]
            
            if len(X_class) < 2:
                continue
            
            results[str(label)] = {
                'class_separation': self.compute_class_separation(X_class, np.zeros(len(X_class))),  # Single class
                'class_compactness': self.compute_class_compactness(X_class, np.zeros(len(X_class))),  # Single class
                'point_density': self.compute_point_density_score(X_class),
                'spatial_distribution': self.compute_spatial_distribution_score(X_class)
            }
        
        return results
