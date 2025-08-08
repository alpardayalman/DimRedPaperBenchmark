"""Utilities for working with image embeddings."""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ImageEmbeddingAnalyzer:
    """Specialized analyzer for image embeddings.
    
    This class provides utilities specifically designed for analyzing
    image embeddings, including visualization, clustering, and
    semantic analysis.
    
    Attributes:
        embeddings: Image embeddings array.
        metadata: Optional metadata about the images.
        scaler: Scaler used for normalization.
    """
    
    def __init__(
        self,
        embeddings: "NDArray[np.floating]",
        metadata: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> None:
        """Initialize the image embedding analyzer.
        
        Args:
            embeddings: Image embeddings of shape (n_images, n_features).
            metadata: Optional DataFrame with image metadata (filenames, labels, etc.).
            normalize: Whether to normalize embeddings.
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self.scaler = None
        
        if normalize:
            self.scaler = StandardScaler()
            self.embeddings = self.scaler.fit_transform(embeddings)
    
    def analyze_embedding_statistics(self) -> Dict[str, float]:
        """Analyze basic statistics of the embeddings.
        
        Returns:
            Dictionary with embedding statistics.
        """
        return {
            'n_images': self.embeddings.shape[0],
            'n_features': self.embeddings.shape[1],
            'mean_norm': np.mean(np.linalg.norm(self.embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(self.embeddings, axis=1)),
            'feature_variance': np.var(self.embeddings, axis=0).mean(),
            'feature_correlation': np.corrcoef(self.embeddings.T).mean(),
            'sparsity': np.mean(self.embeddings == 0),
            'unique_values': len(np.unique(self.embeddings))
        }
    
    def find_similar_images(
        self,
        query_idx: int,
        n_similar: int = 10,
        metric: str = 'cosine'
    ) -> Tuple["NDArray[np.floating]", "NDArray[np.floating]"]:
        """Find most similar images to a query image.
        
        Args:
            query_idx: Index of the query image.
            n_similar: Number of similar images to return.
            metric: Distance metric to use.
            
        Returns:
            Tuple of (indices, similarities) for similar images.
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        query_embedding = self.embeddings[query_idx:query_idx+1]
        
        if metric == 'cosine':
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            # Get indices of most similar (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            similar_scores = similarities[similar_indices]
        else:
            distances = euclidean_distances(query_embedding, self.embeddings).flatten()
            # Get indices of most similar (excluding self)
            similar_indices = np.argsort(distances)[1:n_similar+1]
            similar_scores = 1 / (1 + distances[similar_indices])  # Convert to similarity
        
        return similar_indices, similar_scores
    
    def analyze_semantic_clusters(
        self,
        labels: Optional["NDArray"] = None,
        n_clusters: int = 10
    ) -> Dict[str, Any]:
        """Analyze semantic clustering of image embeddings.
        
        Args:
            labels: Optional ground truth labels.
            n_clusters: Number of clusters for unsupervised analysis.
            
        Returns:
            Dictionary with clustering analysis results.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        results = {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_score(self.embeddings, cluster_labels)
        }
        
        # If ground truth labels are available
        if labels is not None:
            results['adjusted_rand_score'] = adjusted_rand_score(labels, cluster_labels)
            
            # Analyze cluster-label correspondence
            cluster_label_correspondence = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_labels_dist = labels[cluster_mask]
                    unique_labels, counts = np.unique(cluster_labels_dist, return_counts=True)
                    dominant_label = unique_labels[np.argmax(counts)]
                    cluster_label_correspondence[cluster_id] = {
                        'dominant_label': dominant_label,
                        'purity': np.max(counts) / np.sum(counts)
                    }
            results['cluster_label_correspondence'] = cluster_label_correspondence
        
        return results
    
    def visualize_embeddings(
        self,
        method: str = 'umap',
        labels: Optional["NDArray"] = None,
        title: str = "Image Embeddings Visualization",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
        """Create visualization of image embeddings.
        
        Args:
            method: Dimension reduction method ('umap', 'tsne', 'pca').
            labels: Optional labels for coloring.
            title: Plot title.
            figsize: Figure size.
            save_path: Optional path to save the plot.
        """
        from src.dimension_reduction import DimensionReducer
        
        # Apply dimension reduction
        reducer = DimensionReducer()
        reduced_embeddings = reducer.fit_transform(
            self.embeddings, method=method, n_components=2
        )
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        if labels is not None:
            scatter = plt.scatter(
                reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                c=labels, cmap='viridis', alpha=0.7, s=50
            )
            plt.colorbar(scatter, label='Labels')
        else:
            plt.scatter(
                reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                alpha=0.7, s=50
            )
        
        plt.title(title)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_feature_importance(
        self,
        labels: "NDArray",
        method: str = 'random_forest'
    ) -> Dict[str, "NDArray[np.floating]"]:
        """Analyze feature importance for classification.
        
        Args:
            labels: Target labels for classification.
            method: Method for feature importance ('random_forest', 'correlation').
            
        Returns:
            Dictionary with feature importance information.
        """
        if method == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(self.embeddings, labels)
            
            return {
                'feature_importance': rf.feature_importances_,
                'feature_importance_std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
            }
        
        elif method == 'correlation':
            # Compute correlation between each feature and labels
            correlations = []
            for i in range(self.embeddings.shape[1]):
                corr = np.corrcoef(self.embeddings[:, i], labels)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            return {
                'feature_importance': np.array(correlations),
                'feature_importance_std': np.zeros(len(correlations))
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_embedding_report(
        self,
        labels: Optional["NDArray"] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a comprehensive report of embedding analysis.
        
        Args:
            labels: Optional labels for supervised analysis.
            output_path: Optional path to save the report.
            
        Returns:
            Dictionary with comprehensive analysis results.
        """
        report = {
            'basic_statistics': self.analyze_embedding_statistics(),
            'semantic_clusters': self.analyze_semantic_clusters(labels),
        }
        
        if labels is not None:
            report['feature_importance'] = self.analyze_feature_importance(labels)
        
        # Save report if path provided
        if output_path:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            serializable_report = convert_numpy(report)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_report, f, indent=2)
        
        return report


def load_image_embeddings(
    file_path: str,
    format: str = 'numpy'
) -> "NDArray[np.floating]":
    """Load image embeddings from file.
    
    Args:
        file_path: Path to the embeddings file.
        format: File format ('numpy', 'csv', 'json').
        
    Returns:
        Loaded embeddings array.
    """
    if format == 'numpy':
        return np.load(file_path)
    elif format == 'csv':
        return pd.read_csv(file_path).values
    elif format == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return np.array(data)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_image_embeddings(
    embeddings: "NDArray[np.floating]",
    file_path: str,
    format: str = 'numpy'
) -> None:
    """Save image embeddings to file.
    
    Args:
        embeddings: Embeddings array to save.
        file_path: Path to save the embeddings.
        format: File format ('numpy', 'csv', 'json').
    """
    if format == 'numpy':
        np.save(file_path, embeddings)
    elif format == 'csv':
        pd.DataFrame(embeddings).to_csv(file_path, index=False)
    elif format == 'json':
        with open(file_path, 'w') as f:
            json.dump(embeddings.tolist(), f)
    else:
        raise ValueError(f"Unknown format: {format}")


def compare_embedding_sources(
    embeddings_dict: Dict[str, "NDArray[np.floating]"],
    labels: Optional["NDArray"] = None
) -> Dict[str, Dict[str, float]]:
    """Compare different embedding sources.
    
    Args:
        embeddings_dict: Dictionary mapping source names to embeddings.
        labels: Optional labels for supervised evaluation.
        
    Returns:
        Dictionary with comparison results.
    """
    from src.benchmarking import BenchmarkSuite
    
    results = {}
    
    for source_name, embeddings in embeddings_dict.items():
        print(f"Analyzing {source_name} embeddings...")
        
        # Create analyzer
        analyzer = ImageEmbeddingAnalyzer(embeddings)
        
        # Basic statistics
        stats = analyzer.analyze_embedding_statistics()
        
        # Semantic clustering
        clusters = analyzer.analyze_semantic_clusters(labels)
        
        # Feature importance (if labels available)
        feature_importance = None
        if labels is not None:
            feature_importance = analyzer.analyze_feature_importance(labels)
        
        results[source_name] = {
            'statistics': stats,
            'clustering': clusters,
            'feature_importance': feature_importance
        }
    
    return results
