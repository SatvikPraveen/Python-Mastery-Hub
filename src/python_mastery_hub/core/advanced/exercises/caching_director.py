"""Caching director exercise - implement caching with decorators."""


class CachingDirectorExercise:
    """Implement an intelligent caching director using decorators."""

    def __init__(self):
        self.name = "Caching Director Exercise"
        self.description = "Build an intelligent caching system"

    def get_exercise(self):
        return {
            "title": "Caching Director",
            "description": "Create a decorator-based caching system with LRU eviction",
            "difficulty": "advanced",
        }


__all__ = ["CachingDirectorExercise"]
