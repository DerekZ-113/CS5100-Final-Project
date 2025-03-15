"""
AI Models Service

This module handles initialization and access to AI models used in the resume analysis process,
including NER, summarization, and skill classification models.
"""

import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Initialize global model variables
keyword_extractor = None
summarizer = None
tokenizer = None
skill_classifier = None
tfidf = None

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.warning(f"Could not download NLTK data: {str(e)}")
        raise RuntimeError(f"Failed to download NLTK resources: {str(e)}")

def initialize_models():
    """Initialize all AI models required for resume analysis"""
    global keyword_extractor, summarizer, tokenizer, skill_classifier, tfidf
    
    try:
        # Download NLTK resources first
        download_nltk_resources()
        
        # Initialize NER model for keyword extraction
        logger.info("Loading NER model for keyword extraction...")
        keyword_extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        
        # Initialize summarization model
        logger.info("Loading summarization model...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Initialize skill classification model
        logger.info("Loading skill classification model...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        skill_classifier = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        
        # Initialize TF-IDF vectorizer for improved keyword extraction
        logger.info("Setting up TF-IDF vectorizer...")
        tfidf = TfidfVectorizer(
            min_df=2,
            max_df=0.95,
            max_features=200,
            stop_words=stopwords.words('english')
        )
        
        logger.info("All AI models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading AI models: {str(e)}")
        raise RuntimeError(f"Failed to initialize AI models: {str(e)}")

def get_keyword_extractor():
    """Return the keyword extraction model"""
    if keyword_extractor is None:
        raise RuntimeError("Keyword extraction model has not been initialized")
    return keyword_extractor

def get_summarizer():
    """Return the summarization model"""
    if summarizer is None:
        raise RuntimeError("Summarization model has not been initialized")
    return summarizer

def get_skill_classifier():
    """Return the skill classification model and tokenizer"""
    if skill_classifier is None or tokenizer is None:
        raise RuntimeError("Skill classification model has not been initialized")
    return tokenizer, skill_classifier

def get_tfidf_vectorizer():
    """Return the TF-IDF vectorizer"""
    if tfidf is None:
        raise RuntimeError("TF-IDF vectorizer has not been initialized")
    return tfidf

def extract_keywords_with_ner(text, confidence_threshold=0.85):
    """
    Extract keywords from text using the NER model
    
    Args:
        text: The text to extract keywords from
        confidence_threshold: Minimum confidence score to consider a keyword valid
        
    Returns:
        List of extracted keywords
    """
    model = get_keyword_extractor()
    ner_results = model(text)
    
    # Filter keywords by confidence score and length
    extracted_keywords = [
        keyword["word"] for keyword in ner_results 
        if keyword["score"] > confidence_threshold and len(keyword["word"]) > 1
    ]
    
    return extracted_keywords

def summarize_text(text, max_length=500, min_length=100):
    """
    Generate a summary of the input text
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary
        min_length: Minimum length of the summary
        
    Returns:
        Summarized text
    """
    model = get_summarizer()
    
    try:
        # Limit input text length if necessary (model may have max token limit)
        if len(text.split()) > 1000:
            text = " ".join(text.split()[:1000])
        
        summary = model(
            text, 
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            temperature=0.7
        )[0]["summary_text"]
        
        return summary
    except Exception as e:
        logger.error(f"Error in text summarization: {str(e)}")
        return "Could not summarize the text. It may be too long or contain unsupported formatting."

# Create an initialization function to be called during app startup
def setup_ai_services():
    """Initialize all AI services and models"""
    initialize_models()
    logger.info("AI services setup complete")