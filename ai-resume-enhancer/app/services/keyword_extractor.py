"""
Keyword Extractor Service

This module handles extraction of keywords, skills, and important phrases from resume text.
It provides various methods for keyword extraction including NER-based, TF-IDF based,
and pattern-matching approaches.
"""

import re
import logging
from typing import List, Dict, Tuple, Set
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import AI models
from app.services.ai_models import (
    get_keyword_extractor,
    get_tfidf_vectorizer,
    extract_keywords_with_ner
)

# Import skill lists
from app.models.schemas import TECHNICAL_SKILLS, SOFT_SKILLS

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

def extract_keywords_tfidf(text: str, top_n: int = 20) -> List[str]:
    """
    Extract keywords from text using TF-IDF vectorization
    
    Args:
        text: The text to extract keywords from
        top_n: Number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    try:
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english') and len(t) > 2]
        processed_text = " ".join(tokens)
        
        # Get TF-IDF vectorizer
        tfidf = get_tfidf_vectorizer()
        
        # Create a small corpus with the single document
        corpus = [processed_text]
        
        # Fit TF-IDF
        tfidf_matrix = tfidf.fit_transform(corpus)
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        # Get scores
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in sorted_scores[:top_n]]
    except Exception as e:
        logger.warning(f"TF-IDF keyword extraction failed: {str(e)}")
        # Fall back to simple frequency-based extraction
        return extract_keywords_frequency(text, top_n)

def extract_keywords_frequency(text: str, top_n: int = 20) -> List[str]:
    """
    Extract keywords based on word frequency
    
    Args:
        text: The text to extract keywords from
        top_n: Number of keywords to return
        
    Returns:
        List of most frequent words
    """
    try:
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            t for t in tokens 
            if t.isalpha() and 
            t not in stopwords.words('english') and 
            len(t) > 2
        ]
        freq_dist = Counter(filtered_tokens)
        return [word for word, _ in freq_dist.most_common(top_n)]
    except Exception as e:
        logger.error(f"Frequency-based keyword extraction failed: {str(e)}")
        return []

def extract_skills(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract technical and soft skills from text
    
    Args:
        text: The text to extract skills from
        
    Returns:
        Tuple of (technical_skills, soft_skills)
    """
    text = text.lower()
    words = word_tokenize(text)
    
    # Create bigrams and trigrams for multi-word skill detection
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    
    technical_skills = []
    soft_skills = []
    
    # Check for unigrams, bigrams, and trigrams in skills lists
    all_terms = words + bigrams + trigrams
    
    for term in all_terms:
        if term in TECHNICAL_SKILLS:
            technical_skills.append(term)
        elif term in SOFT_SKILLS:
            soft_skills.append(term)
    
    # Check for partial matches for technical skills
    # This helps catch variations like "Python programming" when "python" is in the skills list
    for skill in TECHNICAL_SKILLS:
        if skill not in technical_skills and skill in text:
            technical_skills.append(skill)
    
    # Check for partial matches for soft skills
    for skill in SOFT_SKILLS:
        if skill not in soft_skills and skill in text:
            soft_skills.append(skill)
    
    # Remove duplicates while preserving order
    technical_skills = list(dict.fromkeys(technical_skills))
    soft_skills = list(dict.fromkeys(soft_skills))
    
    return technical_skills, soft_skills

def extract_important_phrases(text: str, min_length: int = 5, max_length: int = 20) -> List[str]:
    """
    Extract important phrases based on capitalization and formatting
    
    Args:
        text: The text to extract phrases from
        min_length: Minimum phrase length
        max_length: Maximum phrase length
        
    Returns:
        List of important phrases
    """
    # Look for phrases that might be section headers or important bullet points
    patterns = [
        # Section headers (capitalized text followed by colon or newline)
        r'(?:^|\n)([A-Z][A-Za-z\s]{' + str(min_length) + ',' + str(max_length) + '})(?::|\n)',
        
        # Bullet points (starting with •, *, -, or number)
        r'(?:^|\n)(?:•|\*|\-|\d+\.)\s+([A-Za-z][A-Za-z\s,]{' + str(min_length) + ',' + str(max_length) + '})',
        
        # All caps phrases (like "PROFESSIONAL EXPERIENCE")
        r'\b([A-Z][A-Z\s]{' + str(min_length) + ',' + str(max_length) + '})\b',
        
        # Action phrases (starting with verb)
        r'(?:^|\n)(?:•|\*|\-|\d+\.)\s+([A-Za-z]ed|[A-Za-z]ing)[A-Za-z\s,]{' + str(min_length-3) + ',' + str(max_length-3) + '}'
    ]
    
    phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phrases.extend([match.strip() for match in matches if match.strip()])
    
    # Remove duplicates while preserving order
    unique_phrases = []
    seen = set()
    for phrase in phrases:
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            unique_phrases.append(phrase)
    
    return unique_phrases[:15]  # Limit to top 15 phrases

def calculate_keyword_frequency(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Calculate frequency of keywords in the text
    
    Args:
        text: The text to analyze
        keywords: List of keywords to count
        
    Returns:
        Dictionary mapping keywords to their frequencies
    """
    words = word_tokenize(text.lower())
    filtered_words = [w for w in words if w.isalpha() and len(w) > 2]
    
    # Count occurrences of keywords
    keyword_freq = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        count = 0
        
        # For multi-word keywords, check if the whole phrase appears
        if ' ' in keyword:
            count = text.lower().count(keyword_lower)
        else:
            count = filtered_words.count(keyword_lower)
        
        if count > 0:
            keyword_freq[keyword] = count
    
    return keyword_freq

def analyze_keywords(text: str) -> Dict[str, any]:
    """
    Comprehensive keyword analysis of text
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with keywords, skills, phrases and frequency data
    """
    try:
        # Extract keywords using NER (high confidence only)
        ner_keywords = extract_keywords_with_ner(text, confidence_threshold=0.85)
        
        # Extract keywords using TF-IDF
        tfidf_keywords = extract_keywords_tfidf(text, top_n=15)
        
        # Combine and deduplicate keywords
        all_keywords = []
        seen = set()
        
        for keyword in ner_keywords + tfidf_keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                all_keywords.append(keyword)
        
        # Limit to top 30 keywords
        all_keywords = all_keywords[:30]
        
        # Extract technical and soft skills
        technical_skills, soft_skills = extract_skills(text)
        
        # Extract important phrases
        important_phrases = extract_important_phrases(text)
        
        # Calculate keyword frequency
        keyword_freq = calculate_keyword_frequency(text, all_keywords + technical_skills + soft_skills)
        
        # Prepare response
        return {
            "keywords": all_keywords,
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "important_phrases": important_phrases,
            "keyword_frequency": keyword_freq
        }
    
    except Exception as e:
        logger.error(f"Error in keyword analysis: {str(e)}")
        # Return a minimal result set on error
        return {
            "keywords": extract_keywords_frequency(text, top_n=10),
            "technical_skills": [],
            "soft_skills": [],
            "important_phrases": [],
            "keyword_frequency": {}
        }