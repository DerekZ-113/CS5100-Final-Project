"""
API Routes Configuration

This module configures and registers all API routes for the AI Resume Enhancer.
It acts as a central registry for all endpoints.
"""

from fastapi import APIRouter
import logging

# Import all endpoint routers
from app.api.endpoints import resume, keywords, improvements, job_matching

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Create main API router
api_router = APIRouter()

# Register all endpoint routers
api_router.include_router(resume.router, prefix="/api/v1")
api_router.include_router(keywords.router, prefix="/api/v1")
api_router.include_router(improvements.router, prefix="/api/v1")
api_router.include_router(job_matching.router, prefix="/api/v1")

# Log registered routes
logger.info("API routes registered")