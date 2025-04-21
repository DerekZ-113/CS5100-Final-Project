"""
Services Package for AI Resume Enhancer

This package contains all service modules that provide core functionality
for resume analysis, keyword extraction, text processing, and job matching.
"""

import logging
import os
import nltk  # Add this import

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

def download_nltk_resources():
    """Download all required NLTK resources"""
    try:
        # Make sure NLTK data directory exists and is writable
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download specific resources
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        
        # Force download of punkt tokenizer data
        # This should ensure the punkt_tab file is available
        try:
            from nltk.tokenize import word_tokenize
            # Force loading to validate installation
            word_tokenize("Test sentence.")
            logger.info("NLTK word_tokenize tested successfully")
        except Exception as e:
            logger.error(f"Error testing word_tokenize: {str(e)}")
            # Try a direct import approach
            try:
                from nltk.data import load
                load('tokenizers/punkt/english.pickle')
                logger.info("NLTK punkt loaded directly")
            except Exception as e2:
                logger.error(f"Error loading punkt directly: {str(e2)}")
                
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK resources: {str(e)}")
        raise

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
        
        # Download NLTK resources first
        download_nltk_resources()
        
        # Initialize AI models next
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
    'initialize_services',
    'download_nltk_resources'
]