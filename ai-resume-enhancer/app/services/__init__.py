"""
Services Package for AI Resume Enhancer

This package contains all service modules that provide core functionality
for resume analysis, keyword extraction, text processing, and job matching.
"""

import logging

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Import key functionality from service modules
from .ai_models import (
    initialize_models,
    extract_keywords_with_ner,
    summarize_text,
    get_keyword_extractor,
    get_summarizer,
    get_tfidf_vectorizer,
    setup_ai_services
)

from .pdf_parser import (
    extract_text_from_pdf,
    clean_extracted_text,
    process_resume_file
)

from .keyword_extractor import (
    extract_keywords_tfidf,
    extract_keywords_frequency,
    extract_skills,
    extract_important_phrases,
    analyze_keywords
)

from .text_processor import (
    generate_improvement_suggestions,
    generate_action_items,
    improve_text_section,
    suggest_resume_improvements
)

# Function to initialize all services
def initialize_services():
    """
    Initialize all services required by the application.
    This should be called during application startup.
    
    Returns:
        bool: True if initialization was successful
    """
    try:
        logger.info("Initializing AI Resume Enhancer services...")
        
        # Initialize AI models first
        setup_ai_services()
        
        logger.info("All services initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise RuntimeError(f"Service initialization failed: {str(e)}")

__all__ = [
    # AI Models
    'initialize_models',
    'extract_keywords_with_ner',
    'summarize_text',
    'get_keyword_extractor',
    'get_summarizer',
    'get_tfidf_vectorizer',
    
    # PDF Parser
    'extract_text_from_pdf',
    'clean_extracted_text',
    'process_resume_file',
    
    # Keyword Extractor
    'extract_keywords_tfidf',
    'extract_keywords_frequency',
    'extract_skills',
    'extract_important_phrases',
    'analyze_keywords',
    
    # Text Processor
    'generate_improvement_suggestions',
    'generate_action_items',
    'improve_text_section',
    'suggest_resume_improvements',
    
    # Service Initialization
    'initialize_services'
]