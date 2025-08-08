"""Main benchmarking suite for dimension reduction algorithms."""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from .structure import StructurePreservation
from .clustering import ClusteringQuality
from .classification import ClassificationQuality
from .visualization import VisualizationQuality

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BenchmarkSuite:
    """Comprehensive benchmarking suite for dimension reduction algorithms.
    
    This class provides a unified interface for evaluating dimension reduction
    algorithms using multiple metrics across different aspects: structure
    preservation, clustering quality, classification performance, and
    visualization quality.
    
    Attributes:
        random_state: Random state for reproducibility.
        structure_metrics: Structure preservation metrics.
        clustering_metrics: Clustering quality metrics.
        classification_metrics: Classification quality metrics.
        visualization_metrics: Visualization quality metrics.
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """Initialize the benchmark suite.
        
        Args:
            random_state: Random state for reproducibility.
        """
        self.random_state = random_state
        self.structure_metrics = StructurePreservation(random_state=random_state)
        self.clustering_metrics = ClusteringQuality(random_state=random_state)
        self.classification_metrics = ClassificationQuality(random_state=random_state)
        self.visualization_metrics = VisualizationQuality(random_state=random_state)
    
    def evaluate_all(
        self,
        X_original: "NDArray[np.floating]",
        embeddings: Dict[str, "NDArray[np.floating]"],
        y_labels: Optional["NDArray"] = None,
        compute_timing: bool = True
    ) -> Dict[str, Any]:
        """Evaluate all embeddings using comprehensive metrics.
        
        Args:
            X_original: Original high-dimensional data.
            embeddings: Dictionary mapping method names to their embeddings.
            y_labels: Target labels for supervised evaluation.
            compute_timing: Whether to compute timing information.
            
        Returns:
            Comprehensive evaluation results.
        """
        results = {
            'structure_preservation': {},
            'clustering_quality': {},
            'classification_quality': {},
            'visualization_quality': {},
            'timing': {},
            'summary': {}
        }
        
        for method_name, embedding in embeddings.items():
            print(f"Evaluating {method_name}...")
            
            # Structure preservation
            results['structure_preservation'][method_name] = self.structure_metrics.compute(
                X_original, embedding
            )
            
            # Clustering quality
            results['clustering_quality'][method_name] = self.clustering_metrics.compute(
                X_original, embedding
            )
            
            # Classification quality (if labels provided)
            if y_labels is not None:
                results['classification_quality'][method_name] = self.classification_metrics.compute(
                    embedding, y_labels
                )
            
            # Visualization quality
            results['visualization_quality'][method_name] = self.visualization_metrics.compute(
                embedding, y_labels
            )
            
            # Timing (if requested)
            if compute_timing:
                results['timing'][method_name] = {
                    'embedding_shape': embedding.shape,
                    'original_shape': X_original.shape,
                    'reduction_ratio': X_original.shape[1] / embedding.shape[1]
                }
        
        # Generate summary
        results['summary'] = self._generate_summary(results, y_labels is not None)
        
        return results
    
    def evaluate_structure_preservation(
        self,
        X_original: "NDArray[np.floating]",
        embeddings: Dict[str, "NDArray[np.floating]"]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate structure preservation for all embeddings.
        
        Args:
            X_original: Original high-dimensional data.
            embeddings: Dictionary mapping method names to their embeddings.
            
        Returns:
            Structure preservation results for each method.
        """
        results = {}
        for method_name, embedding in embeddings.items():
            results[method_name] = self.structure_metrics.compute(X_original, embedding)
        return results
    
    def evaluate_clustering_quality(
        self,
        embeddings: Dict[str, "NDArray[np.floating]"],
        n_clusters: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate clustering quality for all embeddings.
        
        Args:
            embeddings: Dictionary mapping method names to their embeddings.
            n_clusters: Number of clusters to use.
            
        Returns:
            Clustering quality results for each method.
        """
        results = {}
        for method_name, embedding in embeddings.items():
            results[method_name] = self.clustering_metrics.compute(embedding, embedding)
        return results
    
    def evaluate_classification_quality(
        self,
        embeddings: Dict[str, "NDArray[np.floating]"],
        y_labels: "NDArray"
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate classification quality for all embeddings.
        
        Args:
            embeddings: Dictionary mapping method names to their embeddings.
            y_labels: Target labels.
            
        Returns:
            Classification quality results for each method.
        """
        results = {}
        for method_name, embedding in embeddings.items():
            results[method_name] = self.classification_metrics.compute(embedding, y_labels)
        return results
    
    def evaluate_visualization_quality(
        self,
        embeddings: Dict[str, "NDArray[np.floating]"],
        y_labels: Optional["NDArray"] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate visualization quality for all embeddings.
        
        Args:
            embeddings: Dictionary mapping method names to their embeddings.
            y_labels: Target labels for supervised evaluation.
            
        Returns:
            Visualization quality results for each method.
        """
        results = {}
        for method_name, embedding in embeddings.items():
            results[method_name] = self.visualization_metrics.compute(embedding, y_labels)
        return results
    
    def _generate_summary(
        self,
        results: Dict[str, Any],
        has_labels: bool
    ) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results.
        
        Args:
            results: Evaluation results.
            has_labels: Whether labels were provided.
            
        Returns:
            Summary statistics.
        """
        methods = list(results['structure_preservation'].keys())
        
        summary = {
            'methods': methods,
            'best_methods': {},
            'rankings': {},
            'overall_scores': {}
        }
        
        # Find best methods for each category
        summary['best_methods']['structure_preservation'] = self._find_best_method(
            results['structure_preservation'], 'overall_structure_score'
        )
        summary['best_methods']['clustering_quality'] = self._find_best_method(
            results['clustering_quality'], 'clustering_quality_score'
        )
        
        if has_labels:
            summary['best_methods']['classification_quality'] = self._find_best_method(
                results['classification_quality'], 'overall_classification_score'
            )
        
        summary['best_methods']['visualization_quality'] = self._find_best_method(
            results['visualization_quality'], 'overall_visualization_score'
        )
        
        # Generate rankings
        summary['rankings']['structure_preservation'] = self._rank_methods(
            results['structure_preservation'], 'overall_structure_score'
        )
        summary['rankings']['clustering_quality'] = self._rank_methods(
            results['clustering_quality'], 'clustering_quality_score'
        )
        
        if has_labels:
            summary['rankings']['classification_quality'] = self._rank_methods(
                results['classification_quality'], 'overall_classification_score'
            )
        
        summary['rankings']['visualization_quality'] = self._rank_methods(
            results['visualization_quality'], 'overall_visualization_score'
        )
        
        # Compute overall scores
        summary['overall_scores'] = self._compute_overall_scores(results, has_labels)
        
        return summary
    
    def _find_best_method(
        self,
        category_results: Dict[str, Dict[str, float]],
        metric_name: str
    ) -> str:
        """Find the best method for a given category and metric.
        
        Args:
            category_results: Results for a specific category.
            metric_name: Name of the metric to optimize.
            
        Returns:
            Name of the best method.
        """
        best_method = None
        best_score = -float('inf')
        
        for method, metrics in category_results.items():
            if metric_name in metrics and metrics[metric_name] > best_score:
                best_score = metrics[metric_name]
                best_method = method
        
        return best_method or list(category_results.keys())[0]
    
    def _rank_methods(
        self,
        category_results: Dict[str, Dict[str, float]],
        metric_name: str
    ) -> List[str]:
        """Rank methods by a specific metric.
        
        Args:
            category_results: Results for a specific category.
            metric_name: Name of the metric to rank by.
            
        Returns:
            List of method names ranked by performance.
        """
        method_scores = []
        for method, metrics in category_results.items():
            score = metrics.get(metric_name, 0.0)
            method_scores.append((method, score))
        
        # Sort by score (descending)
        method_scores.sort(key=lambda x: x[1], reverse=True)
        return [method for method, _ in method_scores]
    
    def _compute_overall_scores(
        self,
        results: Dict[str, Any],
        has_labels: bool
    ) -> Dict[str, float]:
        """Compute overall scores for each method.
        
        Args:
            results: Evaluation results.
            has_labels: Whether labels were provided.
            
        Returns:
            Overall scores for each method.
        """
        methods = list(results['structure_preservation'].keys())
        overall_scores = {}
        
        for method in methods:
            scores = []
            
            # Structure preservation (weight: 0.3)
            structure_score = results['structure_preservation'][method].get('overall_structure_score', 0.0)
            scores.append(0.3 * structure_score)
            
            # Clustering quality (weight: 0.25)
            clustering_score = results['clustering_quality'][method].get('clustering_quality_score', 0.0)
            scores.append(0.25 * clustering_score)
            
            # Classification quality (weight: 0.25 if labels available)
            if has_labels:
                classification_score = results['classification_quality'][method].get('overall_classification_score', 0.0)
                scores.append(0.25 * classification_score)
            
            # Visualization quality (weight: 0.2)
            visualization_score = results['visualization_quality'][method].get('overall_visualization_score', 0.0)
            scores.append(0.2 * visualization_score)
            
            overall_scores[method] = sum(scores)
        
        return overall_scores
    
    def generate_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Generate a comprehensive report as a pandas DataFrame.
        
        Args:
            results: Evaluation results.
            
        Returns:
            DataFrame with comprehensive results.
        """
        methods = list(results['structure_preservation'].keys())
        
        # Create DataFrame
        report_data = []
        for method in methods:
            row = {'Method': method}
            
            # Structure preservation metrics
            structure = results['structure_preservation'][method]
            row.update({
                'Neighborhood_Preservation': structure.get('neighborhood_preservation', 0.0),
                'Trustworthiness': structure.get('trustworthiness', 0.0),
                'Continuity': structure.get('continuity', 0.0),
                'Stress': structure.get('stress', 0.0),
                'Overall_Structure_Score': structure.get('overall_structure_score', 0.0)
            })
            
            # Clustering quality metrics
            clustering = results['clustering_quality'][method]
            row.update({
                'Silhouette_Score': clustering.get('silhouette_score', 0.0),
                'Calinski_Harabasz': clustering.get('calinski_harabasz_score', 0.0),
                'Davies_Bouldin': clustering.get('davies_bouldin_score', 0.0),
                'Clustering_Quality_Score': clustering.get('clustering_quality_score', 0.0)
            })
            
            # Classification quality metrics (if available)
            if 'classification_quality' in results and method in results['classification_quality']:
                classification = results['classification_quality'][method]
                row.update({
                    'Accuracy': classification.get('accuracy', 0.0),
                    'F1_Score': classification.get('f1_score', 0.0),
                    'Overall_Classification_Score': classification.get('overall_classification_score', 0.0)
                })
            
            # Visualization quality metrics
            visualization = results['visualization_quality'][method]
            row.update({
                'Class_Separation': visualization.get('class_separation', 0.0),
                'Class_Compactness': visualization.get('class_compactness', 0.0),
                'Overall_Visualization_Score': visualization.get('overall_visualization_score', 0.0)
            })
            
            # Overall score
            if 'summary' in results and 'overall_scores' in results['summary']:
                row['Overall_Score'] = results['summary']['overall_scores'].get(method, 0.0)
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to a file.
        
        Args:
            results: Evaluation results.
            filename: Output filename.
        """
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
        
        serializable_results = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
