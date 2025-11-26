"""
Utility modules for Data Structures learning module.
"""

from .best_practices import (
    BEST_PRACTICES,
    get_all_best_practices,
    get_best_practices,
    get_practices_by_difficulty,
    search_best_practices,
)
from .explanations import (
    EXPLANATIONS,
    get_all_explanations,
    get_explanation,
    search_explanations,
)

__all__ = [
    "EXPLANATIONS",
    "get_explanation",
    "get_all_explanations",
    "search_explanations",
    "BEST_PRACTICES",
    "get_best_practices",
    "get_all_best_practices",
    "search_best_practices",
    "get_practices_by_difficulty",
]
