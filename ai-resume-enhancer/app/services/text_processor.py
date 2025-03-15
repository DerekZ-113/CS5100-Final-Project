"""
Text Processor Service

This module handles text processing functions for resume enhancement, including
generating improvement suggestions, summarizing content, and enhancing resume sections.
"""

import re
import logging
from typing import List, Dict, Any
import spacy

# Import AI models for text enhancement
from app.services.ai_models import get_summarizer

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Try to load spaCy model for more advanced NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Could not load spaCy model: {str(e)}. Some functionality will be limited.")
    nlp = None

def generate_improvement_suggestions(text: str) -> List[str]:
    """
    Generate specific improvement suggestions for a resume
    
    Args:
        text: The resume text to analyze
        
    Returns:
        List of specific improvement suggestions
    """
    suggestions = []
    
    # Check for common resume improvement areas
    if len(text.split()) < 200:
        suggestions.append("Your resume seems brief. Consider adding more details about your experience and accomplishments.")
    
    if text.lower().count("i ") > 5:
        suggestions.append("Limit the use of first-person pronouns ('I', 'me', 'my') in your resume.")
    
    # Check for common resume sections
    section_checks = [
        ("education", "Consider adding an Education section if you have relevant academic credentials."),
        ("skills", "Add a dedicated Skills section to highlight your technical and soft skills."),
        ("experience", "Make sure your Experience section details specific accomplishments, not just job duties."),
        ("projects", "Consider adding a Projects section to showcase practical applications of your skills.")
    ]
    
    for section, suggestion in section_checks:
        if not re.search(rf'(?:^|\n)(?:{section}|{section.upper()}|{section.capitalize()})', text, re.IGNORECASE):
            suggestions.append(suggestion)
    
    # Check for action verbs
    action_verbs = [
        "achieved", "improved", "led", "managed", "created", "developed", 
        "implemented", "decreased", "increased", "designed", "launched", 
        "negotiated", "organized", "presented", "researched", "streamlined"
    ]
    
    found_verbs = [verb for verb in action_verbs if verb in text.lower()]
    if len(found_verbs) < 3:
        suggestions.append("Use more strong action verbs (like 'achieved', 'improved', 'developed') to describe your accomplishments.")
    
    # Check for metrics and achievements
    if not re.search(r'\d+%', text) and not re.search(r'\$\d+', text) and not re.search(r'\d+ [a-zA-Z]+', text):
        suggestions.append("Add quantifiable achievements with percentages, dollar amounts, or other metrics to demonstrate your impact.")
    
    # Check for contact information
    contact_patterns = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', "email address"),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "phone number"),
        (r'linkedin\.com\/in\/[A-Za-z0-9_-]+', "LinkedIn profile")
    ]
    
    missing_contacts = []
    for pattern, name in contact_patterns:
        if not re.search(pattern, text):
            missing_contacts.append(name)
    
    if missing_contacts:
        suggestions.append(f"Add your {', '.join(missing_contacts)} to your contact information.")
    
    # Check for formatting consistency
    if text.count('\n\n') > text.count('\n') / 2:
        suggestions.append("Your resume has inconsistent spacing. Consider using consistent spacing between sections.")
    
    return suggestions

def generate_action_items(text: str, keyword_analysis: Dict[str, Any] = None) -> List[str]:
    """
    Generate specific action items for improving a resume
    
    Args:
        text: The resume text
        keyword_analysis: Optional keyword analysis results to inform suggestions
        
    Returns:
        List of actionable steps for resume improvement
    """
    action_items = []
    
    # Basic action items that apply to most resumes
    basic_items = [
        "Ensure your contact information is current and professional",
        "Proofread your resume for grammar and spelling errors",
        "Customize your resume for each job application",
        "Use a clean, professional formatting style",
        "Keep your resume to 1-2 pages depending on experience level"
    ]
    
    # Add basic items to the list
    action_items.extend(basic_items)
    
    # Add skill-specific action items if keyword analysis is available
    if keyword_analysis and 'technical_skills' in keyword_analysis:
        if len(keyword_analysis['technical_skills']) < 5:
            action_items.append("Add more technical skills relevant to your target positions")
        else:
            action_items.append("Organize your technical skills by category for better readability")
    
    # Add content-specific action items based on text analysis
    if len(text.split()) > 700:  # Roughly 1.5 pages
        action_items.append("Consider trimming content to focus on your most relevant experience")
    
    if len(text.split()) < 300:  # Roughly half a page
        action_items.append("Expand your resume with more details about your achievements and skills")
    
    # Add advanced formatting suggestions
    action_items.append("Use bold formatting to highlight key achievements or skills")
    action_items.append("Consider adding a brief professional summary at the beginning of your resume")
    
    return action_items[:7]  # Limit to 7 action items for usability

def improve_text_section(text: str, max_length: int = 500, min_length: int = 100) -> str:
    """
    Generate an improved version of a resume section using AI summarization
    
    Args:
        text: The original text to improve
        max_length: Maximum length of the improved text
        min_length: Minimum length of the improved text
        
    Returns:
        Improved version of the text
    """
    try:
        summarizer = get_summarizer()
        
        # Adjust length parameters based on input text
        summary_length = min(int(len(text) * 0.8), max_length)  # 80% of original or max_length
        min_summary_length = min(int(len(text) * 0.5), min_length)  # 50% of original or min_length
        
        # Generate improved text
        improved_text = summarizer(
            text, 
            max_length=summary_length,
            min_length=min_summary_length,
            do_sample=True,
            temperature=0.7
        )[0]["summary_text"]
        
        return improved_text
    except Exception as e:
        logger.warning(f"Text improvement failed: {str(e)}")
        return text  # Return original text if improvement fails

def analyze_sentence_structure(text: str) -> Dict[str, Any]:
    """
    Analyze sentence structure of resume for more detailed improvements
    
    Args:
        text: Resume text to analyze
        
    Returns:
        Dictionary with sentence analysis metrics
    """
    if not nlp:
        return {"error": "spaCy model not available for sentence analysis"}
    
    try:
        # Parse text with spaCy
        doc = nlp(text)
        
        # Analyze sentences
        sentences = list(doc.sents)
        
        # Calculate metrics
        avg_sentence_length = sum(len(sent) for sent in sentences) / max(len(sentences), 1)
        long_sentences = [str(sent) for sent in sentences if len(sent) > 25]
        passive_voice_count = sum(1 for sent in sentences if _has_passive_voice(sent))
        
        return {
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_length,
            "long_sentences_count": len(long_sentences),
            "passive_voice_count": passive_voice_count,
            "long_sentence_examples": long_sentences[:3]  # Show up to 3 examples
        }
    except Exception as e:
        logger.warning(f"Sentence analysis failed: {str(e)}")
        return {"error": f"Sentence analysis failed: {str(e)}"}

def _has_passive_voice(sentence) -> bool:
    """
    Check if a sentence uses passive voice
    
    Args:
        sentence: spaCy sentence span
        
    Returns:
        True if sentence appears to use passive voice
    """
    # Simple passive voice detection
    # This is a simplified approach and may not catch all passive voice forms
    for token in sentence:
        if token.dep_ == "nsubjpass":
            return True
    return False

def suggest_resume_improvements(text: str, keyword_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Comprehensive resume improvement function
    
    Args:
        text: Resume text to improve
        keyword_analysis: Optional keyword analysis to inform improvements
        
    Returns:
        Dictionary with improvement suggestions and enhanced text
    """
    try:
        # Generate specific improvement suggestions
        suggestions = generate_improvement_suggestions(text)
        
        # Generate action items
        action_items = generate_action_items(text, keyword_analysis)
        
        # Attempt to improve the text
        improved_text = improve_text_section(text)
        
        # Analyze sentence structure
        sentence_analysis = analyze_sentence_structure(text)
        
        # Add additional sentence-specific suggestions
        if "error" not in sentence_analysis:
            if sentence_analysis["passive_voice_count"] > 3:
                suggestions.append("Reduce use of passive voice. Use active voice to make your achievements stand out.")
            
            if sentence_analysis["long_sentences_count"] > 5:
                suggestions.append("Break down long sentences to improve readability.")
        
        return {
            "original_text": text,
            "improved_text": improved_text,
            "suggestions": suggestions,
            "action_items": action_items,
            "sentence_analysis": sentence_analysis
        }
    except Exception as e:
        logger.error(f"Error suggesting improvements: {str(e)}")
        return {
            "original_text": text,
            "improved_text": text,
            "suggestions": ["Could not generate specific suggestions. Please check your resume manually."],
            "action_items": ["Review your resume for clarity and conciseness"],
            "error": str(e)
        }