# File: src/python_mastery_hub/database/__init__.py

"""Database package for Python Mastery Hub.

This package provides database configuration, connection management,
session handling, and utilities for the application.
"""

from .base import Base, create_engine_instance, get_database_url
from .connection import DatabaseManager, get_database_manager
from .session import SessionManager, get_async_session, get_session
from .utils import (
    backup_database,
    check_database_connection,
    create_tables,
    drop_tables,
    get_table_info,
    reset_database,
    restore_database,
)

__all__ = [
    # Base configuration
    "Base",
    "get_database_url",
    "create_engine_instance",
    # Connection management
    "DatabaseManager",
    "get_database_manager",
    # Session management
    "get_session",
    "get_async_session",
    "SessionManager",
    # Utilities
    "create_tables",
    "drop_tables",
    "reset_database",
    "check_database_connection",
    "get_table_info",
    "backup_database",
    "restore_database",
]
