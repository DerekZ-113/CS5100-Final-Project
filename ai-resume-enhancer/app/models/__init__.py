"""
Models package for AI Resume Enhancer.
Contains Pydantic models for API validation and response schemas.
"""

# Import models to expose them at the package level
from .schemas import (
    ResumeAnalysisRequest,
    ResumeJobRequest,
    KeywordResponse,
    MatchResponse,
    ImprovementResponse,
    FileUploadResponse,
    FullAnalysisResponse,
    TECHNICAL_SKILLS,
    SOFT_SKILLS
)

# This allows importing directly from app.models instead of app.models.schemas
# Example: from app.models import ResumeAnalysisRequest