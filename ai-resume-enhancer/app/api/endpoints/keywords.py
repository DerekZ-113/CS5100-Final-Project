"""
Keyword Extraction API Endpoints

This module contains endpoints for extracting keywords, skills, and important phrases
from resume text, providing insights into the resume's content.
"""

import logging
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from app.models.schemas import ResumeAnalysisRequest, KeywordResponse
from app.services.keyword_extractor import (
    analyze_keywords,
    extract_keywords_tfidf,
    extract_skills,
    extract_important_phrases
)

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Create router
router = APIRouter(
    prefix="/keywords",
    tags=["keywords"],
    responses={404: {"description": "Not found"}},
)

# Define response cache to improve performance
response_cache = {}

@router.post("/extract/", response_model=KeywordResponse, status_code=status.HTTP_200_OK)
async def extract_keywords(request: ResumeAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Extract keywords, skills, and important phrases from resume text
    
    - **text**: Resume text to analyze
    
    Returns extracted keywords, skills categorized by type, and frequency data
    """
    try:
        # Check cache first
        cache_key = f"kw_{hash(request.text)}"
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        # Get comprehensive keyword analysis
        result = analyze_keywords(request.text)
        
        # Cache response
        background_tasks.add_task(lambda: response_cache.update({cache_key: result}))
        
        return result
    
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract keywords: {str(e)}"
        )

@router.post("/skills/", status_code=status.HTTP_200_OK)
async def extract_resume_skills(request: ResumeAnalysisRequest):
    """
    Extract only technical and soft skills from resume text
    
    - **text**: Resume text to analyze
    
    Returns lists of technical and soft skills
    """
    try:
        text = request.text
        
        # Extract just the skills
        technical_skills, soft_skills = extract_skills(text)
        
        return {
            "technical_skills": technical_skills,
            "soft_skills": soft_skills
        }
    
    except Exception as e:
        logger.error(f"Error extracting skills: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract skills: {str(e)}"
        )

@router.post("/phrases/", status_code=status.HTTP_200_OK)
async def extract_resume_phrases(request: ResumeAnalysisRequest):
    """
    Extract important phrases from resume text
    
    - **text**: Resume text to analyze
    
    Returns list of important phrases found in the resume
    """
    try:
        text = request.text
        
        # Extract important phrases
        phrases = extract_important_phrases(text)
        
        return {
            "important_phrases": phrases
        }
    
    except Exception as e:
        logger.error(f"Error extracting phrases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract phrases: {str(e)}"
        )

@router.post("/tfidf/", status_code=status.HTTP_200_OK)
async def extract_tfidf_keywords(request: ResumeAnalysisRequest):
    """
    Extract keywords using TF-IDF algorithm
    
    - **text**: Resume text to analyze
    - **top_n**: Number of keywords to extract (optional, default=20)
    
    Returns list of keywords extracted using TF-IDF
    """
    try:
        text = request.text
        top_n = 20  # Default value, could be made a parameter
        
        # Extract keywords with TF-IDF
        keywords = extract_keywords_tfidf(text, top_n)
        
        return {
            "tfidf_keywords": keywords
        }
    
    except Exception as e:
        logger.error(f"Error extracting TF-IDF keywords: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract TF-IDF keywords: {str(e)}"
        )