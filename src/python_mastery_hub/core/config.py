"""
Configuration management for Python Mastery Hub.

Handles environment-specific settings and application configuration.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Application
    app_name: str = "Python Mastery Hub"
    app_version: str = "1.0.0"
    environment: Literal["development", "testing", "production"] = os.getenv(
        "ENVIRONMENT", "development"
    )
    debug: bool = environment == "development"

    # Server
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", 8000))

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    database_echo: bool = environment == "development"
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", 20))
    database_max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", 40))

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    refresh_token_expire_days: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

    # CORS
    cors_origins: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).split(",")
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]

    # Email
    email_enabled: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", 587))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    from_email: str = os.getenv("FROM_EMAIL", "noreply@pythonmasteryhub.com")

    # Logging
    log_level: str = os.getenv(
        "LOG_LEVEL", "INFO" if environment == "production" else "DEBUG"
    )
    log_format: str = "json" if environment == "production" else "text"

    # Code Execution
    code_execution_timeout: int = int(os.getenv("CODE_EXECUTION_TIMEOUT", 30))
    max_code_length: int = int(os.getenv("MAX_CODE_LENGTH", 10000))

    # Features
    enable_code_execution: bool = (
        os.getenv("ENABLE_CODE_EXECUTION", "true").lower() == "true"
    )
    enable_progress_tracking: bool = (
        os.getenv("ENABLE_PROGRESS_TRACKING", "true").lower() == "true"
    )
    enable_social_features: bool = (
        os.getenv("ENABLE_SOCIAL_FEATURES", "true").lower() == "true"
    )

    class Config:
        """Pydantic settings configuration."""

        case_sensitive = False
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings: Application configuration instance
    """
    return Settings()


def get_environment() -> str:
    """Get current environment name.

    Returns:
        str: Environment name (development, testing, production)
    """
    return get_settings().environment


def is_development() -> bool:
    """Check if running in development environment.

    Returns:
        bool: True if development environment
    """
    return get_settings().environment == "development"


def is_production() -> bool:
    """Check if running in production environment.

    Returns:
        bool: True if production environment
    """
    return get_settings().environment == "production"


def is_testing() -> bool:
    """Check if running in testing environment.

    Returns:
        bool: True if testing environment
    """
    return get_settings().environment == "testing"
