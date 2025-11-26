"""
Algorithms Learning Module.

Comprehensive coverage of fundamental algorithms including sorting, searching,
dynamic programming, graph algorithms, and algorithmic thinking patterns.
"""

from .. import LearningModule
from .algorithmic_patterns import AlgorithmicPatterns
from .base import Algorithms
from .dynamic_programming import DynamicProgramming
from .graph_algorithms import GraphAlgorithms
from .searching import SearchingAlgorithms
from .sorting import SortingAlgorithms

__all__ = [
    "Algorithms",
    "SortingAlgorithms",
    "SearchingAlgorithms",
    "DynamicProgramming",
    "GraphAlgorithms",
    "AlgorithmicPatterns",
]
