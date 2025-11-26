# src/python_mastery_hub/utils/__init__.py
"""
Python Mastery Hub Utils Package

Core utility modules providing shared functionality across the application
including progress tracking, achievements, logging, validation, and file handling.

Optional modules require additional dependencies:
- security_utils: requires 'bcrypt' (pip install bcrypt)
- metrics_collector: requires 'psutil' (pip install psutil)  
- data_exporters: Excel support requires 'openpyxl' (pip install openpyxl)
"""

from .achievement_engine import (
    Achievement,
    AchievementCategory,
    AchievementEngine,
    AchievementTier,
)
from .cache_manager import (
    CacheManager,
    FileCache,
    MemoryCache,
    cache_clear,
    cache_delete,
    cache_get,
    cache_set,
    cache_stats,
    cached,
    get_cache,
)
from .code_execution import (
    CodeValidator,
    ExecutionError,
    ExecutionResult,
    SafeCodeExecutor,
    SecurityViolation,
    TestRunner,
    execute_code,
    run_code_tests,
    validate_code,
)
from .data_exporters import CSVExporter, DataExporter
from .data_exporters import ExportConfig as DataExportConfig
from .data_exporters import (
    HTMLReportExporter,
    JSONExporter,
    ZipExporter,
    create_exporter,
    export_learning_data,
    get_supported_formats,
)
from .email_templates import (
    EmailPreferences,
    EmailRenderer,
    EmailTemplate,
    EmailTemplateManager,
    create_email_manager,
    render_achievement_email,
    render_welcome_email,
)
from .file_handlers import (
    ConfigFileHandler,
    CSVFileHandler,
    DirectoryHandler,
    FileError,
    JSONFileHandler,
    SafeFileHandler,
    YAMLFileHandler,
    read_csv,
    read_json,
    read_text,
    read_yaml,
    write_csv,
    write_json,
    write_text,
    write_yaml,
)
from .formatters import (
    ColorCodes,
    TextFormatter,
    format_duration,
    format_json,
    format_percentage,
    format_progress_bar,
    format_table,
    format_time_ago,
    formatter,
)
from .logging_config import (
    PerformanceLogger,
    cleanup_old_logs,
    create_module_logger,
    get_logger,
    get_logging_status,
    log_performance,
    log_user_action,
    set_log_level,
    setup_logging,
    timed_operation,
)

# Core utilities - always available
from .progress_calculator import (
    LearningSession,
    ModuleProgress,
    ProgressCalculator,
    TopicProgress,
)
from .validators import (
    FormValidator,
    NumberValidator,
    StringValidator,
    ValidationError,
    Validator,
    email_validator,
    number_validator,
    sanitize_string,
    string_validator,
    validate_email,
    validate_python_code,
    validate_url,
)

# Core exports - always available
__all__ = [
    # Progress tracking
    "ProgressCalculator",
    "TopicProgress",
    "ModuleProgress",
    "LearningSession",
    # Achievement system
    "AchievementEngine",
    "Achievement",
    "AchievementCategory",
    "AchievementTier",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "set_log_level",
    "log_performance",
    "log_user_action",
    "create_module_logger",
    "PerformanceLogger",
    "timed_operation",
    "cleanup_old_logs",
    "get_logging_status",
    # Validation
    "ValidationError",
    "Validator",
    "StringValidator",
    "NumberValidator",
    "FormValidator",
    "validate_email",
    "validate_url",
    "validate_python_code",
    "sanitize_string",
    "string_validator",
    "number_validator",
    "email_validator",
    # Formatting
    "TextFormatter",
    "ColorCodes",
    "formatter",
    "format_duration",
    "format_time_ago",
    "format_percentage",
    "format_progress_bar",
    "format_table",
    "format_json",
    # File handling
    "SafeFileHandler",
    "JSONFileHandler",
    "YAMLFileHandler",
    "CSVFileHandler",
    "ConfigFileHandler",
    "DirectoryHandler",
    "FileError",
    "read_text",
    "write_text",
    "read_json",
    "write_json",
    "read_yaml",
    "write_yaml",
    "read_csv",
    "write_csv",
    # Code execution
    "SafeCodeExecutor",
    "CodeValidator",
    "TestRunner",
    "ExecutionResult",
    "ExecutionError",
    "SecurityViolation",
    "execute_code",
    "validate_code",
    "run_code_tests",
    # Email templates
    "EmailTemplateManager",
    "EmailRenderer",
    "EmailTemplate",
    "EmailPreferences",
    "create_email_manager",
    "render_welcome_email",
    "render_achievement_email",
    # Caching
    "CacheManager",
    "MemoryCache",
    "FileCache",
    "cached",
    "get_cache",
    "cache_get",
    "cache_set",
    "cache_delete",
    "cache_clear",
    "cache_stats",
    # Data export
    "DataExporter",
    "CSVExporter",
    "JSONExporter",
    "HTMLReportExporter",
    "ZipExporter",
    "DataExportConfig",
    "create_exporter",
    "export_learning_data",
    "get_supported_formats",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Python Mastery Hub"
__description__ = "Comprehensive utilities for the Python Mastery Hub learning platform"


# Optional module access functions
def get_security_utils():
    """
    Get security utilities module (requires bcrypt).

    Returns:
        security_utils module

    Raises:
        ImportError: If bcrypt is not installed
    """
    try:
        from . import security_utils

        return security_utils
    except ImportError as e:
        raise ImportError(
            "Security utilities require bcrypt. Install with: pip install bcrypt"
        ) from e


def get_metrics_collector():
    """
    Get metrics collector module (requires psutil).

    Returns:
        metrics_collector module

    Raises:
        ImportError: If psutil is not installed
    """
    try:
        from . import metrics_collector

        return metrics_collector
    except ImportError as e:
        raise ImportError(
            "Metrics collection requires psutil. Install with: pip install psutil"
        ) from e


def get_excel_exporter():
    """
    Get Excel export functionality (requires openpyxl).

    Returns:
        ExcelExporter class

    Raises:
        ImportError: If openpyxl is not installed
    """
    try:
        from .data_exporters import ExcelExporter

        return ExcelExporter
    except ImportError as e:
        raise ImportError(
            "Excel export requires openpyxl. Install with: pip install openpyxl"
        ) from e


def check_dependencies():
    """
    Check which optional dependencies are available.

    Returns:
        Dict[str, bool]: Availability status of optional features
    """
    features = {}

    # Check bcrypt for security
    try:
        import bcrypt

        features["security"] = True
    except ImportError:
        features["security"] = False

    # Check psutil for metrics
    try:
        import psutil

        features["metrics"] = True
    except ImportError:
        features["metrics"] = False

    # Check openpyxl for Excel export
    try:
        import openpyxl

        features["excel_export"] = True
    except ImportError:
        features["excel_export"] = False

    return features


# Convenience imports for optional modules (use with caution)
def import_security():
    """Import security utilities with clear error message if unavailable."""
    return get_security_utils()


def import_metrics():
    """Import metrics collector with clear error message if unavailable."""
    return get_metrics_collector()
