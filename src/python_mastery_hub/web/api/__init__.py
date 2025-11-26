"""
API Package - REST API Endpoints

Contains all API route definitions for the Python Mastery Hub web application.
"""

from . import admin, auth, exercises, modules, progress

# Export API routers
__all__ = [
    "auth",
    "modules",
    "progress",
    "exercises",
    "admin",
]
