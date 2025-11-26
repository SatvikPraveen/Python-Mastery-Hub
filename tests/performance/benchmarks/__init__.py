# tests/performance/benchmarks/__init__.py
"""
Performance benchmarks package for the Python learning platform.
Contains benchmark tests for measuring and tracking performance metrics
over time to detect regressions and improvements.
"""

__version__ = "1.0.0"
__author__ = "Python Learning Platform Team"

from .api_benchmarks import APIBenchmarks

# Import main benchmark classes for easy access
from .benchmark_runner import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .code_execution_benchmarks import CodeExecutionBenchmarks
from .database_benchmarks import DatabaseBenchmarks

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkConfig",
    "CodeExecutionBenchmarks",
    "DatabaseBenchmarks",
    "APIBenchmarks",
]
