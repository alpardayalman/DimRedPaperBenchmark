"""Dimension Reduction Toolkit.

A comprehensive toolkit for dimension reduction techniques with benchmarking capabilities.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@hb.com"

from .dimension_reduction import DimensionReducer
from .benchmarking import BenchmarkSuite

__all__ = ["DimensionReducer", "BenchmarkSuite"]
