"""
Resume Improvements API Endpoints

This module contains endpoints for generating improvement suggestions
and enhancing resume content.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from app.models.schemas import ResumeAnalysisRequest, ImprovementResponse
from app.services.text_processor import suggest_resume_improvements
from app.services.keyword_extractor import analyze_keywords
import logging

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Create router
router = APIRouter(
    prefix="/improvements",
    tags=["improvements"],
    responses={404: {"description": "Not found"}},
)

@router.post("/suggest/", response_model=ImprovementResponse, status_code=status.HTTP_200_OK)
async def suggest_improvements(request: ResumeAnalysisRequest):
    """
    Generate improvement suggestions for resume text
    
    - **text**: Resume text to analyze
    
    Returns improvement suggestions and enhanced text
    """
    try:
        text = request.text
        
        # Get improvement suggestions
        improvement_result = suggest_resume_improvements(text)
        
        return {
            "original_text": text,
            "improved_text": improvement_result.get("improved_text", text),
            "suggestions": improvement_result.get("suggestions", []),
            "action_items": improvement_result.get("action_items", [])
        }
    
    except Exception as e:
        logger.error(f"Error suggesting improvements: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to suggest improvements: {str(e)}"
        )

@router.post("/enhanced-suggest/", response_model=ImprovementResponse, status_code=status.HTTP_200_OK)
async def enhanced_improvements(request: ResumeAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Generate comprehensive improvement suggestions with keyword analysis
    
    - **text**: Resume text to analyze
    
    Returns improvement suggestions informed by keyword analysis
    """
    try:
        text = request.text
        
        # Perform keyword analysis first
        keyword_analysis = analyze_keywords(text)
        
        # Get enhanced improvement suggestions with keyword context
        improvement_result = suggest_resume_improvements(text, keyword_analysis)
        
        return {
            "original_text": text,
            "improved_text": improvement_result.get("improved_text", text),
            "suggestions": improvement_result.get("suggestions", []),
            "action_items": improvement_result.get("action_items", [])
        }
    
    except Exception as e:
        logger.error(f"Error generating enhanced improvements: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate enhanced improvements: {str(e)}"
        )