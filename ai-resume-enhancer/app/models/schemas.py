from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator

# Define Pydantic models for request/response validation
class ResumeAnalysisRequest(BaseModel):
    """Request model for resume analysis endpoints"""
    text: str = Field(..., min_length=10, description="The resume text to analyze")
    
    @validator('text')
    def text_must_be_meaningful(cls, v):
        if len(v.strip()) < 100:
            raise ValueError("Resume text is too short to provide meaningful analysis")
        return v

class ResumeJobRequest(BaseModel):
    """Request model for resume-job comparison endpoint"""
    resume_text: str = Field(..., min_length=10, description="Resume text")
    job_text: str = Field(..., min_length=10, description="Job description text")
    
    @validator('resume_text', 'job_text')
    def text_must_be_meaningful(cls, v):
        if len(v.strip()) < 50:
            raise ValueError("Text is too short to provide meaningful analysis")
        return v

class KeywordResponse(BaseModel):
    """Response model for keyword analysis"""
    keywords: List[str]
    technical_skills: List[str]
    soft_skills: List[str]
    important_phrases: List[str]
    keyword_frequency: Dict[str, int]

class MatchResponse(BaseModel):
    """Response model for resume-job matching"""
    match_score: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    skill_gap_analysis: Dict[str, List[str]]
    improvement_suggestions: List[str]

class ImprovementResponse(BaseModel):
    """Response model for resume improvement suggestions"""
    original_text: str
    improved_text: str
    suggestions: List[str]
    action_items: List[str]

class FileUploadResponse(BaseModel):
    """Response model for file upload endpoint"""
    filename: str
    content_type: str
    file_size: int
    extracted_text: str
    text_preview: str

class FullAnalysisResponse(BaseModel):
    """Response model for the comprehensive analysis endpoint"""
    keywords_analysis: KeywordResponse
    improvement_suggestions: ImprovementResponse
    analysis_timestamp: str

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