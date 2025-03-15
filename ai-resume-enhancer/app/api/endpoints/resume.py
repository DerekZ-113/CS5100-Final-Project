"""
Resume Processing API Endpoints

This module contains endpoints for uploading, processing, and analyzing resumes.
It serves as the entry point for the resume enhancement workflow.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from app.models.schemas import ResumeAnalysisRequest, FileUploadResponse, FullAnalysisResponse
from app.services.pdf_parser import process_resume_file
from app.services.keyword_extractor import analyze_keywords
from app.services.text_processor import suggest_resume_improvements
from datetime import datetime

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Create router
router = APIRouter(
    prefix="/resume",
    tags=["resume"],
    responses={404: {"description": "Not found"}},
)

# Define response cache to improve performance
response_cache = {}

@router.post("/upload/", response_model=FileUploadResponse, status_code=status.HTTP_200_OK)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and process a resume file (PDF or TXT)
    
    - **file**: Resume file to upload (PDF or TXT format only)
    
    Returns extracted text and basic file information
    """
    try:
        # Validate file size
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Process based on file type
        if file.content_type == "application/pdf":
            result = process_resume_file(file_content, "application/pdf")
        elif file.content_type == "text/plain":
            result = process_resume_file(file_content, "text/plain")
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type. Please upload a PDF or TXT file."
            )
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(file_content),
            "extracted_text": result["extracted_text"],
            "text_preview": result["text_preview"]
        }
    
    except ValueError as e:
        # Handle expected value errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        # Log and handle unexpected errors
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process resume: {str(e)}"
        )

@router.post("/process-text/", response_model=FileUploadResponse, status_code=status.HTTP_200_OK)
async def process_resume_text(request: ResumeAnalysisRequest):
    """
    Process plain text resume
    
    - **text**: Resume text to process
    
    Returns processed text and preview
    """
    try:
        text = request.text
        
        # Generate preview
        preview_length = min(500, len(text))
        text_preview = text[:preview_length] + ("..." if len(text) > preview_length else "")
        
        return {
            "filename": "text_input.txt",
            "content_type": "text/plain",
            "file_size": len(text.encode('utf-8')),
            "extracted_text": text,
            "text_preview": text_preview
        }
    
    except Exception as e:
        logger.error(f"Error processing resume text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process resume text: {str(e)}"
        )

@router.post("/analyze-full/", response_model=FullAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_full_resume(request: ResumeAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Comprehensive resume analysis endpoint
    
    - **text**: Resume text to analyze
    
    Combines keyword analysis and improvement suggestions in a single call
    """
    try:
        # Check cache first
        cache_key = f"full_{hash(request.text)}"
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        text = request.text
        
        # Get keywords analysis
        keywords_analysis = analyze_keywords(text)
        
        # Get improvement suggestions using keyword context
        improvements_response = suggest_resume_improvements(text, keywords_analysis)
        
        # Format as improvement response
        improvement_result = {
            "original_text": text,
            "improved_text": improvements_response.get("improved_text", text),
            "suggestions": improvements_response.get("suggestions", []),
            "action_items": improvements_response.get("action_items", [])
        }
        
        # Combine responses
        result = {
            "keywords_analysis": keywords_analysis,
            "improvement_suggestions": improvement_result,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Cache response
        background_tasks.add_task(lambda: response_cache.update({cache_key: result}))
        
        return result
    
    except Exception as e:
        logger.error(f"Error in full resume analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete full resume analysis: {str(e)}"
        )

# Periodic cache cleanup could be added as a background task in the main app