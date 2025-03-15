"""
Endpoints Package

This package contains all API endpoint modules organized by functional area.
Each module defines a router with related endpoints for a specific domain.
"""

# Import all endpoint modules to make them available
from . import resume, keywords, improvements, job_matching

# This allows importing the endpoint modules directly from the package
# Example: from app.api.endpoints import resume, keywords

__all__ = ['resume', 'keywords', 'improvements', 'job_matching']