"""Clustering quality metrics for benchmarking dimension reduction."""

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.mixture import GaussianMixture
from .metrics import BenchmarkMetrics

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClusteringQuality(BenchmarkMetrics):
    """Clustering quality metrics for dimension reduction evaluation.
    
    This class provides comprehensive metrics for evaluating how well
    dimension reduction algorithms preserve clustering structure and
    enable effective clustering in the reduced space.
    
    Attributes:
        random_state: Random state for reproducibility.
        n_clusters: Default number of clusters for clustering algorithms.
    """
    
    def __init__(self, random_state: Optional[int] = None, n_clusters: int = 5) -> None:
        """Initialize clustering quality metrics.
        
        Args:
            random_state: Random state for reproducibility.
            n_clusters: Default number of clusters for clustering algorithms.
        """
        super().__init__(random_state)
        self.n_clusters = n_clusters
    
    def compute(self, X_original: "NDArray[np.floating]", X_reduced: "NDArray[np.floating]") -> Dict[str, float]:
        """Compute all clustering quality metrics.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            
        Returns:
            Dictionary of metric names and values.
        """
        return {
            'silhouette_score': self.compute_silhouette_score(X_reduced),
            'calinski_harabasz_score': self.compute_calinski_harabasz_score(X_reduced),
            'davies_bouldin_score': self.compute_davies_bouldin_score(X_reduced),
            'clustering_quality_score': self.compute_clustering_quality_score(X_reduced),
            'cluster_separation_score': self.compute_cluster_separation_score(X_reduced),
            'cluster_compactness_score': self.compute_cluster_compactness_score(X_reduced)
        }
    
    def compute_silhouette_score(
        self,
        X: "NDArray[np.floating]",
        n_clusters: Optional[int] = None
    ) -> float:
        """Compute silhouette score for clustering quality.
        
        The silhouette score measures how similar an object is to its own
        cluster compared to other clusters. Higher values indicate better
        clustering quality.
        
        Args:
            X: Data to cluster.
            n_clusters: Number of clusters to use.
            
        Returns:
            Silhouette score (-1 to 1, higher is better).
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if len(X) < n_clusters + 1:
            return 0.0
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            return silhouette_score(X, labels)
        except Exception:
            return 0.0
    
    def compute_calinski_harabasz_score(
        self,
        X: "NDArray[np.floating]",
        n_clusters: Optional[int] = None
    ) -> float:
        """Compute Calinski-Harabasz score for clustering quality.
        
        This score measures the ratio of between-cluster dispersion and
        within-cluster dispersion. Higher values indicate better clustering.
        
        Args:
            X: Data to cluster.
            n_clusters: Number of clusters to use.
            
        Returns:
            Calinski-Harabasz score (higher is better).
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if len(X) < n_clusters + 1:
            return 0.0
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            return calinski_harabasz_score(X, labels)
        except Exception:
            return 0.0
    
    def compute_davies_bouldin_score(
        self,
        X: "NDArray[np.floating]",
        n_clusters: Optional[int] = None
    ) -> float:
        """Compute Davies-Bouldin score for clustering quality.
        
        This score measures the average similarity measure of each cluster
        with its most similar cluster. Lower values indicate better clustering.
        
        Args:
            X: Data to cluster.
            n_clusters: Number of clusters to use.
            
        Returns:
            Davies-Bouldin score (lower is better).
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if len(X) < n_clusters + 1:
            return float('inf')
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            return davies_bouldin_score(X, labels)
        except Exception:
            return float('inf')
    
    def compute_clustering_quality_score(
        self,
        X: "NDArray[np.floating]",
        n_clusters: Optional[int] = None
    ) -> float:
        """Compute overall clustering quality score.
        
        This combines multiple clustering metrics to provide a comprehensive
        assessment of clustering quality.
        
        Args:
            X: Data to cluster.
            n_clusters: Number of clusters to use.
            
        Returns:
            Overall clustering quality score (0-1, higher is better).
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if len(X) < n_clusters + 1:
            return 0.0
        
        try:
            # Compute individual scores
            silhouette = self.compute_silhouette_score(X, n_clusters)
            calinski_harabasz = self.compute_calinski_harabasz_score(X, n_clusters)
            davies_bouldin = self.compute_davies_bouldin_score(X, n_clusters)
            
            # Normalize scores to 0-1 range
            silhouette_norm = (silhouette + 1) / 2  # Convert from [-1, 1] to [0, 1]
            calinski_norm = min(calinski_harabasz / 1000, 1.0)  # Normalize to reasonable range
            davies_norm = max(0, 1 - davies_bouldin / 10)  # Convert to 0-1 (lower is better)
            
            # Combine scores
            return (silhouette_norm + calinski_norm + davies_norm) / 3
        except Exception:
            return 0.0
    
    def compute_cluster_separation_score(
        self,
        X: "NDArray[np.floating]",
        n_clusters: Optional[int] = None
    ) -> float:
        """Compute cluster separation score.
        
        This measures how well-separated the clusters are from each other.
        
        Args:
            X: Data to cluster.
            n_clusters: Number of clusters to use.
            
        Returns:
            Cluster separation score (0-1, higher is better).
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if len(X) < n_clusters + 1:
            return 0.0
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            
            # Compute average distance between cluster centers
            center_distances = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    center_distances.append(np.linalg.norm(centers[i] - centers[j]))
            
            if not center_distances:
                return 0.0
            
            # Normalize by data range
            data_range = np.max(X) - np.min(X)
            if data_range == 0:
                return 0.0
            
            return min(np.mean(center_distances) / data_range, 1.0)
        except Exception:
            return 0.0
    
    def compute_cluster_compactness_score(
        self,
        X: "NDArray[np.floating]",
        n_clusters: Optional[int] = None
    ) -> float:
        """Compute cluster compactness score.
        
        This measures how compact the clusters are within themselves.
        
        Args:
            X: Data to cluster.
            n_clusters: Number of clusters to use.
            
        Returns:
            Cluster compactness score (0-1, higher is better).
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        if len(X) < n_clusters + 1:
            return 0.0
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            
            # Compute average within-cluster distance
            within_cluster_distances = []
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    distances = np.linalg.norm(cluster_points - centers[i], axis=1)
                    within_cluster_distances.extend(distances)
            
            if not within_cluster_distances:
                return 0.0
            
            # Normalize by data range
            data_range = np.max(X) - np.min(X)
            if data_range == 0:
                return 0.0
            
            # Convert to compactness score (lower distances = higher compactness)
            avg_distance = np.mean(within_cluster_distances)
            return max(0, 1 - avg_distance / data_range)
        except Exception:
            return 0.0
    
    def compute_clustering_comparison(
        self,
        X_original: "NDArray[np.floating]",
        X_reduced: "NDArray[np.floating]",
        true_labels: Optional["NDArray"] = None,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare clustering quality between original and reduced spaces.
        
        Args:
            X_original: Original high-dimensional data.
            X_reduced: Reduced low-dimensional data.
            true_labels: True cluster labels if available.
            n_clusters: Number of clusters to use.
            
        Returns:
            Dictionary comparing clustering metrics between spaces.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        results = {
            'original_space': self.compute(X_original, X_original),  # Self-comparison
            'reduced_space': self.compute(X_reduced, X_reduced)      # Self-comparison
        }
        
        if true_labels is not None:
            # Add external validation metrics
            results['external_validation'] = self.compute_external_clustering_metrics(
                X_reduced, true_labels
            )
        
        return results
    
    def compute_external_clustering_metrics(
        self,
        X: "NDArray[np.floating]",
        true_labels: "NDArray"
    ) -> Dict[str, float]:
        """Compute external clustering validation metrics.
        
        Args:
            X: Data to cluster.
            true_labels: True cluster labels.
            
        Returns:
            Dictionary of external validation metrics.
        """
        try:
            n_clusters = len(np.unique(true_labels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            predicted_labels = kmeans.fit_predict(X)
            
            return {
                'adjusted_rand_score': adjusted_rand_score(true_labels, predicted_labels),
                'normalized_mutual_info_score': normalized_mutual_info_score(true_labels, predicted_labels),
                'homogeneity_score': homogeneity_score(true_labels, predicted_labels),
                'completeness_score': completeness_score(true_labels, predicted_labels),
                'v_measure_score': v_measure_score(true_labels, predicted_labels)
            }
        except Exception:
            return {
                'adjusted_rand_score': 0.0,
                'normalized_mutual_info_score': 0.0,
                'homogeneity_score': 0.0,
                'completeness_score': 0.0,
                'v_measure_score': 0.0
            }
    
    def compute_multiple_clustering_algorithms(
        self,
        X: "NDArray[np.floating]",
        n_clusters: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute clustering quality using multiple algorithms.
        
        Args:
            X: Data to cluster.
            n_clusters: Number of clusters to use.
            
        Returns:
            Dictionary mapping algorithm names to their clustering metrics.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        results = {}
        
        # K-means
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            results['kmeans'] = {
                'silhouette_score': silhouette_score(X, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X, labels),
                'davies_bouldin_score': davies_bouldin_score(X, labels)
            }
        except Exception:
            results['kmeans'] = {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
        
        # Gaussian Mixture
        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
            labels = gmm.fit_predict(X)
            results['gaussian_mixture'] = {
                'silhouette_score': silhouette_score(X, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X, labels),
                'davies_bouldin_score': davies_bouldin_score(X, labels)
            }
        except Exception:
            results['gaussian_mixture'] = {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
        
        # Agglomerative Clustering
        try:
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agg.fit_predict(X)
            results['agglomerative'] = {
                'silhouette_score': silhouette_score(X, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X, labels),
                'davies_bouldin_score': davies_bouldin_score(X, labels)
            }
        except Exception:
            results['agglomerative'] = {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
        
        return results
