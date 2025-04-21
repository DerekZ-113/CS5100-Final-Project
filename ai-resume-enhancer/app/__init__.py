"""
AI Resume Enhancer Application

This module initializes and configures the FastAPI application instance,
including middleware, exception handlers, and API routes.
"""

import logging
import nltk  # Add this import
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import asyncio

# Import API router
from app.api import api_router

# Import services initialization
from app.services import initialize_services

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai-resume-enhancer")

# Create application factory
def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Returns:
        Configured FastAPI application instance
    """
    # Initialize FastAPI with metadata
    app = FastAPI(
        title="AI Resume Enhancer API",
        description="API for analyzing resumes, extracting keywords, and providing improvement suggestions",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router)
    
    # Configure custom exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with a clean JSON response"""
        errors = []
        for error in exc.errors():
            error_msg = error.get("msg", "Validation error")
            error_loc = " -> ".join(str(loc) for loc in error.get("loc", []))
            errors.append(f"{error_loc}: {error_msg}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Validation error", "errors": errors}
        )
    
    # Set up startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on application startup"""
        try:
            logger.info("Starting AI Resume Enhancer API")
            
            # Download required NLTK resources
            try:
                logger.info("Downloading required NLTK resources...")
                nltk.download('punkt', quiet=False)
                nltk.download('stopwords', quiet=False)
                logger.info("NLTK resources downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download NLTK resources: {str(e)}")
                # Continue execution as some functionality might still work
            
            # Initialize services (AI models, etc.)
            initialize_services()
            
            # Start periodic cache cleanup
            asyncio.create_task(periodic_cache_cleanup())
            
            logger.info("Application startup complete")
        except Exception as e:
            logger.error(f"Startup error: {str(e)}")
            # In production, you might want to exit the app if startup fails
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on application shutdown"""
        logger.info("Shutting down AI Resume Enhancer API")
    
    # Add a health check endpoint
    @app.get("/api/health", tags=["health"])
    async def health_check():
        """Check if the API is running"""
        return {"status": "healthy", "version": "1.0.0"}
    
    # Return the configured app
    return app

# Cache cleanup function
async def periodic_cache_cleanup():
    """Periodically clean up response caches to prevent memory issues"""
    while True:
        try:
            # Clean up every hour
            await asyncio.sleep(3600)
            
            # Import here to avoid circular imports
            from app.api.endpoints.resume import response_cache as resume_cache
            from app.api.endpoints.keywords import response_cache as keywords_cache
            from app.api.endpoints.job_matching import response_cache as job_matching_cache
            
            # Clear caches
            resume_cache.clear()
            keywords_cache.clear()
            job_matching_cache.clear()
            
            logger.info("Response caches cleared")
        except Exception as e:
            logger.error(f"Error in cache cleanup: {str(e)}")

# Expose the application instance
app = create_app()