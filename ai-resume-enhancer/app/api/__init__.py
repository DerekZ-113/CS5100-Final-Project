"""
API Package

This package contains all API-related modules including route definitions,
endpoint handlers, and request/response processing logic.
"""

# Import the main router to expose it at the package level
from app.api.routes import api_router

# This allows importing the router directly from app.api
# Example: from app.api import api_router

__all__ = ['api_router']