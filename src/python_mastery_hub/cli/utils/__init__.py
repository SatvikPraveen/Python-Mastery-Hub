"""
CLI Utilities Package

Common utilities for the Python Mastery Hub CLI including
colors, progress bars, and input validation.
"""

from . import colors, input_validation, progress_bar

# Export utility modules
__all__ = [
    "colors",
    "progress_bar",
    "input_validation",
]

# Convenience imports for common utilities
from .colors import (
    Colors,
    print_code_block,
    print_error,
    print_header,
    print_info,
    print_subheader,
    print_success,
    print_warning,
)
from .input_validation import (
    InputValidator,
    ValidationError,
    choice_validator,
    exercise_validator,
    integer_validator,
    module_validator,
    prompt_with_validation,
    validate_and_suggest,
)
from .progress_bar import (
    ProgressBar,
    SpinnerProgress,
    animated_loading,
    show_module_progress,
    show_progress,
)
