"""
Web Development Learning Module.

Comprehensive coverage of web development with Python including Flask, FastAPI,
REST APIs, WebSockets, authentication, database integration, and deployment.
"""

from .. import LearningModule
from .core import WebDevelopment
from .examples import (
    auth_examples,
    database_examples,
    fastapi_examples,
    flask_examples,
    rest_api_examples,
    websocket_examples,
)
from .exercises import (
    flask_blog_exercise,
    jwt_auth_exercise,
    microservice_exercise,
    rest_api_exercise,
    websocket_chat_exercise,
)

__all__ = [
    "WebDevelopment",
    "flask_examples",
    "fastapi_examples",
    "rest_api_examples",
    "websocket_examples",
    "auth_examples",
    "database_examples",
    "rest_api_exercise",
    "websocket_chat_exercise",
    "jwt_auth_exercise",
    "flask_blog_exercise",
    "microservice_exercise",
]
