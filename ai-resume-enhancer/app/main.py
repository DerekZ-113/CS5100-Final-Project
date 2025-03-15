"""
AI Resume Enhancer API - Main Entry Point

This module serves as the entry point for running the FastAPI application.
It imports the configured app and runs it using Uvicorn.
"""

import uvicorn
import argparse
import logging
from app import app

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the AI Resume Enhancer API")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the server on"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    return parser.parse_args()

def main():
    """Run the FastAPI application"""
    args = parse_args()
    
    # Log startup information
    logger.info(f"Starting AI Resume Enhancer API on {args.host}:{args.port}")
    logger.info(f"Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    logger.info(f"API documentation: http://{args.host}:{args.port}/api/docs")
    
    # Run the application with Uvicorn
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()