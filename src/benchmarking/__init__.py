"""Benchmarking framework for dimension reduction algorithms."""

from .metrics import BenchmarkMetrics
from .structure import StructurePreservation
from .clustering import ClusteringQuality
from .classification import ClassificationQuality
from .visualization import VisualizationQuality
from .main import BenchmarkSuite

__all__ = [
    "BenchmarkMetrics",
    "StructurePreservation", 
    "ClusteringQuality",
    "ClassificationQuality",
    "VisualizationQuality",
    "BenchmarkSuite",
]
