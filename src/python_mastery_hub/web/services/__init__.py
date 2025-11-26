# Location: src/python_mastery_hub/web/services/__init__.py

"""
Web Services Package

Contains service layer components for business logic, authentication,
code execution, progress tracking, and external integrations.
"""

from .auth_service import AuthService, PasswordService, TokenService
from .code_executor import CodeExecutionError, CodeExecutor, ExecutionResult
from .email_service import EmailProvider, EmailService, EmailTemplate
from .progress_service import ProgressService, ProgressTracker

__all__ = [
    # Authentication services
    "AuthService",
    "TokenService",
    "PasswordService",
    # Code execution services
    "CodeExecutor",
    "ExecutionResult",
    "CodeExecutionError",
    # Progress tracking services
    "ProgressService",
    "ProgressTracker",
    # Email services
    "EmailService",
    "EmailTemplate",
    "EmailProvider",
]
