"""Classification quality metrics for benchmarking dimension reduction."""

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from .metrics import BenchmarkMetrics

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClassificationQuality(BenchmarkMetrics):
    """Classification quality metrics for dimension reduction evaluation.
    
    This class provides comprehensive metrics for evaluating how well
    dimension reduction algorithms preserve classification-relevant
    information in the reduced space.
    
    Attributes:
        random_state: Random state for reproducibility.
        cv_folds: Number of cross-validation folds.
        classifiers: List of classifiers to use for evaluation.
    """
    
    def __init__(self, random_state: Optional[int] = None, cv_folds: int = 5) -> None:
        """Initialize classification quality metrics.
        
        Args:
            random_state: Random state for reproducibility.
            cv_folds: Number of cross-validation folds.
        """
        super().__init__(random_state)
        self.cv_folds = cv_folds
        self.classifiers = [
            ('logistic_regression', LogisticRegression(random_state=random_state, max_iter=1000)),
            ('random_forest', RandomForestClassifier(random_state=random_state, n_estimators=100)),
            ('svm', SVC(random_state=random_state, probability=True))
        ]
    
    def compute(self, X: "NDArray[np.floating]", y: "NDArray") -> Dict[str, float]:
        """Compute all classification quality metrics.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            
        Returns:
            Dictionary of metric names and values.
        """
        return {
            'accuracy': self.compute_accuracy(X, y),
            'f1_score': self.compute_f1_score(X, y),
            'precision': self.compute_precision(X, y),
            'recall': self.compute_recall(X, y),
            'roc_auc': self.compute_roc_auc(X, y),
            'overall_classification_score': self.compute_overall_classification_score(X, y),
            'classifier_performance': self.compute_classifier_performance(X, y)
        }
    
    def compute_accuracy(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray",
        classifier_name: str = 'logistic_regression'
    ) -> float:
        """Compute classification accuracy using cross-validation.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            classifier_name: Name of the classifier to use.
            
        Returns:
            Mean accuracy across cross-validation folds.
        """
        classifier = self._get_classifier(classifier_name)
        if classifier is None:
            return 0.0
        
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
            return np.mean(scores)
        except Exception:
            return 0.0
    
    def compute_f1_score(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray",
        classifier_name: str = 'logistic_regression'
    ) -> float:
        """Compute F1 score using cross-validation.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            classifier_name: Name of the classifier to use.
            
        Returns:
            Mean F1 score across cross-validation folds.
        """
        classifier = self._get_classifier(classifier_name)
        if classifier is None:
            return 0.0
        
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(classifier, X, y, cv=cv, scoring='f1_weighted')
            return np.mean(scores)
        except Exception:
            return 0.0
    
    def compute_precision(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray",
        classifier_name: str = 'logistic_regression'
    ) -> float:
        """Compute precision using cross-validation.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            classifier_name: Name of the classifier to use.
            
        Returns:
            Mean precision across cross-validation folds.
        """
        classifier = self._get_classifier(classifier_name)
        if classifier is None:
            return 0.0
        
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(classifier, X, y, cv=cv, scoring='precision_weighted')
            return np.mean(scores)
        except Exception:
            return 0.0
    
    def compute_recall(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray",
        classifier_name: str = 'logistic_regression'
    ) -> float:
        """Compute recall using cross-validation.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            classifier_name: Name of the classifier to use.
            
        Returns:
            Mean recall across cross-validation folds.
        """
        classifier = self._get_classifier(classifier_name)
        if classifier is None:
            return 0.0
        
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(classifier, X, y, cv=cv, scoring='recall_weighted')
            return np.mean(scores)
        except Exception:
            return 0.0
    
    def compute_roc_auc(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray",
        classifier_name: str = 'logistic_regression'
    ) -> float:
        """Compute ROC AUC using cross-validation.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            classifier_name: Name of the classifier to use.
            
        Returns:
            Mean ROC AUC across cross-validation folds.
        """
        classifier = self._get_classifier(classifier_name)
        if classifier is None:
            return 0.0
        
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(classifier, X, y, cv=cv, scoring='roc_auc_ovr_weighted')
            return np.mean(scores)
        except Exception:
            return 0.0
    
    def compute_overall_classification_score(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> float:
        """Compute overall classification quality score.
        
        This combines multiple classification metrics to provide a comprehensive
        assessment of classification quality.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            
        Returns:
            Overall classification quality score (0-1, higher is better).
        """
        try:
            # Compute individual scores
            accuracy = self.compute_accuracy(X, y)
            f1 = self.compute_f1_score(X, y)
            precision = self.compute_precision(X, y)
            recall = self.compute_recall(X, y)
            roc_auc = self.compute_roc_auc(X, y)
            
            # Combine scores (equal weights)
            return (accuracy + f1 + precision + recall + roc_auc) / 5
        except Exception:
            return 0.0
    
    def compute_classifier_performance(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> Dict[str, float]:
        """Compute performance across multiple classifiers.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            
        Returns:
            Dictionary mapping classifier names to their accuracy scores.
        """
        results = {}
        
        for classifier_name, _ in self.classifiers:
            try:
                accuracy = self.compute_accuracy(X, y, classifier_name)
                results[classifier_name] = accuracy
            except Exception:
                results[classifier_name] = 0.0
        
        return results
    
    def _get_classifier(self, classifier_name: str):
        """Get classifier by name.
        
        Args:
            classifier_name: Name of the classifier.
            
        Returns:
            Classifier instance or None if not found.
        """
        for name, classifier in self.classifiers:
            if name == classifier_name:
                return classifier
        return None
    
    def compute_classification_by_class(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray",
        classifier_name: str = 'logistic_regression'
    ) -> Dict[str, Dict[str, float]]:
        """Compute classification metrics for each class.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            classifier_name: Name of the classifier to use.
            
        Returns:
            Dictionary mapping class labels to their classification metrics.
        """
        classifier = self._get_classifier(classifier_name)
        if classifier is None:
            return {}
        
        unique_labels = np.unique(y)
        results = {}
        
        for label in unique_labels:
            try:
                # Create binary classification problem for this class
                y_binary = (y == label).astype(int)
                
                # Compute metrics for this class
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                
                accuracy = np.mean(cross_val_score(classifier, X, y_binary, cv=cv, scoring='accuracy'))
                f1 = np.mean(cross_val_score(classifier, X, y_binary, cv=cv, scoring='f1'))
                precision = np.mean(cross_val_score(classifier, X, y_binary, cv=cv, scoring='precision'))
                recall = np.mean(cross_val_score(classifier, X, y_binary, cv=cv, scoring='recall'))
                
                results[str(label)] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }
            except Exception:
                results[str(label)] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                }
        
        return results
    
    def compute_feature_importance_analysis(
        self,
        X: "NDArray[np.floating]",
        y: "NDArray"
    ) -> Dict[str, "NDArray[np.floating]"]:
        """Compute feature importance analysis using Random Forest.
        
        Args:
            X: Reduced data for classification.
            y: Target labels.
            
        Returns:
            Dictionary with feature importance information.
        """
        try:
            rf = RandomForestClassifier(random_state=self.random_state, n_estimators=100)
            rf.fit(X, y)
            
            return {
                'feature_importance': rf.feature_importances_,
                'feature_importance_std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
            }
        except Exception:
            return {
                'feature_importance': np.zeros(X.shape[1]),
                'feature_importance_std': np.zeros(X.shape[1])
            }
