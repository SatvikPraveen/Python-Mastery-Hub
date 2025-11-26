"""
Exercise implementations for the Advanced Python module.

This package contains standalone exercise implementations that demonstrate
advanced Python concepts through practical, hands-on projects.
"""

from .caching_director import CachingDecoratorExercise
from .file_pipeline import FilePipelineExercise
from .orm_metaclass import ORMMetaclassExercise
from .transaction_manager import TransactionManagerExercise

__all__ = [
    "CachingDecoratorExercise",
    "FilePipelineExercise",
    "TransactionManagerExercise",
    "ORMMetaclassExercise",
]
