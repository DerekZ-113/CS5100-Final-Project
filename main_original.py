from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
import uvicorn
import re
import os
import json
import logging
from datetime import datetime
import asyncio
import io
import PyPDF2
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize FastAPI with metadata
app = FastAPI(
    title="AI Resume Enhancer API",
    description="API for analyzing resumes, extracting keywords, and providing improvement suggestions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai-resume-enhancer")

# Download nltk data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {str(e)}")

# Initialize AI models
try:
    keyword_extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Add additional skill extractor model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    skill_classifier = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    # Initialize TF-IDF vectorizer for improved keyword extraction
    tfidf = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        max_features=200,
        stop_words=stopwords.words('english')
    )
    
    logger.info("AI models loaded successfully")
except Exception as e:
    logger.error(f"Error loading AI models: {str(e)}")
    raise

# Define response cache to improve performance
response_cache = {}

# Define Pydantic models for request/response validation
class ResumeAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The resume text to analyze")
    
    @validator('text')
    def text_must_be_meaningful(cls, v):
        if len(v.strip()) < 100:
            raise ValueError("Resume text is too short to provide meaningful analysis")
        return v

class ResumeJobRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="Resume text")
    job_text: str = Field(..., min_length=10, description="Job description text")
    
    @validator('resume_text', 'job_text')
    def text_must_be_meaningful(cls, v):
        if len(v.strip()) < 50:
            raise ValueError("Text is too short to provide meaningful analysis")
        return v

class KeywordResponse(BaseModel):
    keywords: List[str]
    technical_skills: List[str]
    soft_skills: List[str]
    important_phrases: List[str]
    keyword_frequency: Dict[str, int]

class MatchResponse(BaseModel):
    match_score: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    skill_gap_analysis: Dict[str, List[str]]
    improvement_suggestions: List[str]

class ImprovementResponse(BaseModel):
    original_text: str
    improved_text: str
    suggestions: List[str]
    action_items: List[str]

# Common skills lists
TECHNICAL_SKILLS = set([
    "python", "java", "javascript", "c++", "sql", "react", "angular", "node.js", "aws",
    "docker", "kubernetes", "machine learning", "data science", "tensorflow", "pytorch",
    "tableau", "power bi", "excel", "git", "agile", "scrum", "jenkins", "ci/cd", "azure",
    "google cloud", "typescript", "html", "css", "rest api", "spring", "django", "flask",
    "hadoop", "spark", "mysql", "postgresql", "mongodb", "nosql", "linux", "unix", "bash",
    "powershell", "ruby", "php", "swift", "kotlin", "r", "scala", "rust", "go", "golang"
])

SOFT_SKILLS = set([
    "communication", "leadership", "teamwork", "problem solving", "critical thinking",
    "decision making", "time management", "adaptability", "flexibility", "creativity",
    "interpersonal skills", "conflict resolution", "emotional intelligence", "negotiation",
    "presentation skills", "attention to detail", "organization", "planning", "multitasking",
    "collaboration", "self-motivation", "work ethic", "analytical skills", "customer service"
])

# Helper functions
def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes with enhanced error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if not text.strip():
            # Try alternative extraction if standard extraction fails
            text = ""
            for page in pdf_reader.pages:
                page_text = extract_text_fallback(page)
                if page_text:
                    text += page_text + "\n\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError("Could not extract text from PDF. The file may be corrupted or password-protected.")

def extract_text_fallback(page):
    """Fallback method for extracting text from PDF pages"""
    # This is a placeholder for a more robust extraction method
    # In a complete implementation, you might use tools like pdfminer.six or PyMuPDF
    return page.extract_text() or ""

def extract_keywords_tfidf(text, top_n=20):
    """Extract keywords using TF-IDF"""
    try:
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]
        processed_text = " ".join(tokens)
        
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

def extract_keywords_frequency(text, top_n=20):
    """Extract keywords based on frequency"""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english') and len(t) > 2]
    freq_dist = Counter(filtered_tokens)
    return [word for word, _ in freq_dist.most_common(top_n)]

def extract_skills(text):
    """Extract technical and soft skills from text"""
    words = word_tokenize(text.lower())
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    
    technical_skills = []
    soft_skills = []
    
    # Check for unigrams and bigrams in skills lists
    all_terms = words + bigrams
    for term in all_terms:
        if term in TECHNICAL_SKILLS:
            technical_skills.append(term)
        elif term in SOFT_SKILLS:
            soft_skills.append(term)
    
    # Remove duplicates while preserving order
    technical_skills = list(dict.fromkeys(technical_skills))
    soft_skills = list(dict.fromkeys(soft_skills))
    
    return technical_skills, soft_skills

def extract_important_phrases(text, min_length=5, max_length=20):
    """Extract important phrases based on capitals and formatting"""
    # Look for phrases that might be section headers or important bullet points
    patterns = [
        r'(?:^|\n)([A-Z][A-Za-z\s]{' + str(min_length) + ',' + str(max_length) + '})(?::|\n)',  # Section headers
        r'(?:^|\n)(?:•|\*|\-|\d+\.)\s+([A-Za-z][A-Za-z\s,]{' + str(min_length) + ',' + str(max_length) + '})'  # Bullet points
    ]
    
    phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phrases.extend([match.strip() for match in matches])
    
    return phrases[:15]  # Limit to top 15 phrases

def generate_improvement_suggestions(text):
    """Generate specific improvement suggestions for the resume"""
    suggestions = []
    
    # Check for common resume improvement areas
    if len(text.split()) < 200:
        suggestions.append("Your resume seems brief. Consider adding more details about your experience and accomplishments.")
    
    if text.lower().count("i ") > 5:
        suggestions.append("Limit the use of first-person pronouns ('I', 'me', 'my') in your resume.")
    
    if not re.search(r'(?:^|\n)(?:EDUCATION|Education)', text):
        suggestions.append("Consider adding an Education section if you have relevant academic credentials.")
    
    if not re.search(r'(?:^|\n)(?:SKILLS|Skills)', text):
        suggestions.append("Add a dedicated Skills section to highlight your technical and soft skills.")
    
    # Check for action verbs
    action_verbs = ["achieved", "improved", "led", "managed", "created", "developed", "implemented", "decreased", "increased"]
    found_verbs = [verb for verb in action_verbs if verb in text.lower()]
    if len(found_verbs) < 3:
        suggestions.append("Use more strong action verbs (like 'achieved', 'improved', 'developed') to describe your accomplishments.")
    
    # Check for metrics and achievements
    if not re.search(r'\d+%', text) and not re.search(r'\$\d+', text):
        suggestions.append("Add quantifiable achievements with percentages, dollar amounts, or other metrics to demonstrate your impact.")
    
    return suggestions

# API endpoints
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint that confirms the API is running"""
    return {
        "message": "AI Resume Enhancer API is Running!",
        "version": "1.0.0",
        "endpoints": [
            "/upload_resume/",
            "/analyze_keywords/",
            "/suggest_improvements/",
            "/compare_resume_to_job/",
            "/analyze_full_resume/"
        ]
    }

@app.post("/upload_resume/", status_code=status.HTTP_200_OK)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and process a resume file (PDF or TXT)
    
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
            text = extract_text_from_pdf(file_content)
        elif file.content_type == "text/plain":
            text = file_content.decode("utf-8")
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type. Please upload a PDF or TXT file."
            )
        
        # Validate extracted text
        if not text or len(text) < 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract sufficient text from the file. The file may be empty or corrupt."
            )
        
        # Return text with preview
        preview_length = min(500, len(text))
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(file_content),
            "extracted_text": text,
            "text_preview": text[:preview_length] + ("..." if len(text) > preview_length else "")
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
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

@app.post("/analyze_keywords/", response_model=KeywordResponse, status_code=status.HTTP_200_OK)
async def analyze_keywords(request: ResumeAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Extract keywords, skills, and important phrases from resume text
    
    Uses multiple extraction methods including NER, TF-IDF, and pattern matching
    """
    try:
        # Check cache first
        cache_key = f"kw_{hash(request.text)}"
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        text = request.text
        
        # Extract keywords using transformer model (high confidence only)
        ner_results = keyword_extractor(text)
        extracted_ner_keywords = [
            keyword["word"] for keyword in ner_results 
            if keyword["score"] > 0.85 and len(keyword["word"]) > 1
        ]
        
        # Extract keywords using TF-IDF
        tfidf_keywords = extract_keywords_tfidf(text, top_n=15)
        
        # Combine and deduplicate keywords
        all_keywords = list(dict.fromkeys(extracted_ner_keywords + tfidf_keywords))[:30]
        
        # Extract technical and soft skills
        technical_skills, soft_skills = extract_skills(text)
        
        # Extract important phrases
        important_phrases = extract_important_phrases(text)
        
        # Calculate keyword frequency
        words = word_tokenize(text.lower())
        filtered_words = [w for w in words if w.isalpha() and w not in stopwords.words('english') and len(w) > 2]
        keyword_freq = {k: filtered_words.count(k.lower()) for k in all_keywords if k.lower() in filtered_words}
        
        # Prepare response
        response = {
            "keywords": all_keywords,
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "important_phrases": important_phrases,
            "keyword_frequency": keyword_freq
        }
        
        # Cache response
        background_tasks.add_task(lambda: response_cache.update({cache_key: response}))
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing keywords: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze keywords: {str(e)}"
        )

@app.post("/suggest_improvements/", response_model=ImprovementResponse, status_code=status.HTTP_200_OK)
async def suggest_improvements(request: ResumeAnalysisRequest):
    """
    Generate improvement suggestions for resume text
    
    Uses AI summarization and rule-based suggestions
    """
    try:
        text = request.text
        
        # Generate specific improvement suggestions
        suggestions = generate_improvement_suggestions(text)
        
        # Use AI to generate improved text suggestions
        try:
            summary_length = min(int(len(text) * 0.8), 500)  # 80% of original or max 500 tokens
            min_length = min(int(len(text) * 0.5), 200)  # 50% of original or min 200 tokens
            
            improved_text = summarizer(
                text, 
                max_length=summary_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7
            )[0]["summary_text"]
        except Exception as e:
            logger.warning(f"Summarization failed: {str(e)}")
            improved_text = "Could not generate improved text. Please try with a different resume section."
        
        # Generate specific action items
        action_items = [
            "Highlight your most relevant accomplishments at the top of your resume",
            "Use industry-specific keywords relevant to the jobs you're applying for",
            "Ensure your contact information is current and professional",
            "Remove outdated or irrelevant experience",
            "Have someone proofread your resume for grammar and spelling errors"
        ]
        
        return {
            "original_text": text,
            "improved_text": improved_text,
            "suggestions": suggestions,
            "action_items": action_items
        }
    
    except Exception as e:
        logger.error(f"Error suggesting improvements: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to suggest improvements: {str(e)}"
        )

@app.post("/compare_resume_to_job/", response_model=MatchResponse, status_code=status.HTTP_200_OK)
async def compare_resume_to_job(request: ResumeJobRequest):
    """
    Compare resume text to job description
    
    Analyzes keyword matching, skill gaps, and provides match score and improvement suggestions
    """
    try:
        # Extract keywords from both texts
        def extract_keywords(text, min_score=0.85):
            # Extract with NER
            ner_results = keyword_extractor(text)
            ner_keywords = set(
                keyword["word"].lower() for keyword in ner_results 
                if keyword["score"] > min_score and len(keyword["word"]) > 1
            )
            
            # Extract with TF-IDF
            tfidf_keywords = set(extract_keywords_tfidf(text, top_n=20))
            
            # Combine keywords
            all_keywords = ner_keywords.union(tfidf_keywords)
            
            return all_keywords
        
        # Extract technical and soft skills
        resume_tech_skills, resume_soft_skills = extract_skills(request.resume_text)
        job_tech_skills, job_soft_skills = extract_skills(request.job_text)
        
        # Extract general keywords
        resume_keywords = extract_keywords(request.resume_text)
        job_keywords = extract_keywords(request.job_text)
        
        # Combine all keywords and skills for matching
        resume_all = resume_keywords.union(set(k.lower() for k in resume_tech_skills + resume_soft_skills))
        job_all = job_keywords.union(set(k.lower() for k in job_tech_skills + job_soft_skills))
        
        # Find matches and missing keywords
        matched_keywords = resume_all.intersection(job_all)
        missing_keywords = job_all - resume_all
        
        # Calculate comprehensive match score (with higher weight for technical skills)
        if len(job_all) == 0:
            match_score = 0
        else:
            # Basic match percentage
            basic_match = len(matched_keywords) / len(job_all) * 100
            
            # Technical skills match (weighted more heavily)
            job_tech_set = set(s.lower() for s in job_tech_skills)
            resume_tech_set = set(s.lower() for s in resume_tech_skills)
            tech_match = len(resume_tech_set.intersection(job_tech_set)) / max(len(job_tech_set), 1) * 100
            
            # Calculate weighted score (60% technical, 40% overall)
            match_score = (tech_match * 0.6) + (basic_match * 0.4)
            match_score = round(match_score, 2)
        
        # Skill gap analysis
        skill_gaps = {
            "technical_skills": list(set(s.lower() for s in job_tech_skills) - set(s.lower() for s in resume_tech_skills)),
            "soft_skills": list(set(s.lower() for s in job_soft_skills) - set(s.lower() for s in resume_soft_skills))
        }
        
        # Generate improvement suggestions based on match
        improvement_suggestions = []
        
        if match_score < 50:
            improvement_suggestions.append("Your resume shows low compatibility with this job posting. Consider significant revisions to highlight relevant experience and skills.")
        elif match_score < 75:
            improvement_suggestions.append("Your resume shows moderate compatibility with this job. Adding more relevant keywords could improve your match score.")
        
        # Add specific skill gap suggestions
        if skill_gaps["technical_skills"]:
            tech_gaps = ", ".join(skill_gaps["technical_skills"][:5])
            improvement_suggestions.append(f"Add these missing technical skills to your resume if you have them: {tech_gaps}")
        
        if skill_gaps["soft_skills"]:
            soft_gaps = ", ".join(skill_gaps["soft_skills"][:3])
            improvement_suggestions.append(f"Consider incorporating these soft skills in your resume: {soft_gaps}")
        
        # Add general improvement suggestion
        improvement_suggestions.append("Tailor your resume to match key terms from the job description, especially in the skills and experience sections.")
        
        return {
            "match_score": match_score,
            "matched_keywords": list(matched_keywords),
            "missing_keywords": list(missing_keywords),
            "skill_gap_analysis": skill_gaps,
            "improvement_suggestions": improvement_suggestions
        }
    
    except Exception as e:
        logger.error(f"Error comparing resume to job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare resume to job: {str(e)}"
        )

@app.post("/analyze_full_resume/", status_code=status.HTTP_200_OK)
async def analyze_full_resume(request: ResumeAnalysisRequest):
    """
    Comprehensive resume analysis endpoint
    
    Combines keyword analysis and improvement suggestions in a single call
    """
    try:
        # Get keywords analysis
        keywords_response = await analyze_keywords(request, BackgroundTasks())
        
        # Get improvement suggestions
        improvements_response = await suggest_improvements(request)
        
        # Combine responses
        return {
            "keywords_analysis": keywords_response,
            "improvement_suggestions": improvements_response,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in full resume analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete full resume analysis: {str(e)}"
        )

# Periodic cache cleanup
@app.on_event("startup")
async def startup_event():
    async def cleanup_cache():
        while True:
            try:
                # Clear cache every hour
                await asyncio.sleep(3600)
                response_cache.clear()
                logger.info("Response cache cleared")
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
    
    asyncio.create_task(cleanup_cache())
    logger.info("API startup complete")

# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)