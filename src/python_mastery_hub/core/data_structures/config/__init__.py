"""
Configuration module for Data Structures learning module.
"""

from .topics import (
    TOPICS_CONFIG,
    get_difficulty_levels,
    get_topic_dependencies,
    get_topic_order,
)

__all__ = [
    "TOPICS_CONFIG",
    "get_topic_order",
    "get_topic_dependencies",
    "get_difficulty_levels",
]
