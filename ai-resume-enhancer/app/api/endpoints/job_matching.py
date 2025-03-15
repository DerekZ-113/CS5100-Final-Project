"""
Job Matching API Endpoints

This module contains endpoints for comparing resumes against job descriptions,
calculating match scores, and identifying skill gaps.
"""

import logging
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from app.models.schemas import ResumeJobRequest, MatchResponse
from app.services.keyword_extractor import analyze_keywords, extract_skills

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

# Create router
router = APIRouter(
    prefix="/job-matching",
    tags=["job-matching"],
    responses={404: {"description": "Not found"}},
)

# Define response cache to improve performance
response_cache = {}

@router.post("/compare/", response_model=MatchResponse, status_code=status.HTTP_200_OK)
async def compare_resume_to_job(request: ResumeJobRequest, background_tasks: BackgroundTasks):
    """
    Compare resume text to job description
    
    - **resume_text**: Resume text to analyze
    - **job_text**: Job description text to compare against
    
    Returns match score, matched keywords, missing keywords, and improvement suggestions
    """
    try:
        # Check cache first
        cache_key = f"match_{hash(request.resume_text + request.job_text)}"
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        # Extract keywords from both texts
        resume_analysis = analyze_keywords(request.resume_text)
        job_analysis = analyze_keywords(request.job_text)
        
        # Create sets for comparison
        resume_keywords = set(k.lower() for k in resume_analysis.get("keywords", []))
        job_keywords = set(k.lower() for k in job_analysis.get("keywords", []))
        
        resume_tech_skills = set(s.lower() for s in resume_analysis.get("technical_skills", []))
        job_tech_skills = set(s.lower() for s in job_analysis.get("technical_skills", []))
        
        resume_soft_skills = set(s.lower() for s in resume_analysis.get("soft_skills", []))
        job_soft_skills = set(s.lower() for s in job_analysis.get("soft_skills", []))
        
        # Combine all keywords and skills for matching
        resume_all = resume_keywords.union(resume_tech_skills).union(resume_soft_skills)
        job_all = job_keywords.union(job_tech_skills).union(job_soft_skills)
        
        # Find matches and missing keywords
        matched_keywords = list(resume_all.intersection(job_all))
        missing_keywords = list(job_all - resume_all)
        
        # Calculate comprehensive match score
        if len(job_all) == 0:
            match_score = 0
        else:
            # Basic match percentage
            basic_match = len(resume_all.intersection(job_all)) / len(job_all) * 100
            
            # Technical skills match (weighted more heavily)
            if len(job_tech_skills) > 0:
                tech_match = len(resume_tech_skills.intersection(job_tech_skills)) / len(job_tech_skills) * 100
            else:
                tech_match = 100
            
            # Calculate weighted score (60% technical, 40% overall)
            match_score = (tech_match * 0.6) + (basic_match * 0.4)
            match_score = round(match_score, 2)
        
        # Skill gap analysis
        skill_gaps = {
            "technical_skills": list(job_tech_skills - resume_tech_skills),
            "soft_skills": list(job_soft_skills - resume_soft_skills)
        }
        
        # Generate improvement suggestions based on match
        improvement_suggestions = []
        
        if match_score < 50:
            improvement_suggestions.append("Your resume shows low compatibility with this job posting. Consider significant revisions to highlight relevant experience and skills.")
        elif match_score < 75:
            improvement_suggestions.append("Your resume shows moderate compatibility with this job. Adding more relevant keywords could improve your match score.")
        else:
            improvement_suggestions.append("Your resume shows strong compatibility with this job posting.")
        
        # Add specific skill gap suggestions
        if skill_gaps["technical_skills"]:
            tech_gaps = ", ".join(skill_gaps["technical_skills"][:5])
            improvement_suggestions.append(f"Add these missing technical skills to your resume if you have them: {tech_gaps}")
        
        if skill_gaps["soft_skills"]:
            soft_gaps = ", ".join(skill_gaps["soft_skills"][:3])
            improvement_suggestions.append(f"Consider incorporating these soft skills in your resume: {soft_gaps}")
        
        # Add general improvement suggestion
        improvement_suggestions.append("Tailor your resume to match key terms from the job description, especially in the skills and experience sections.")
        
        # Prepare response
        result = {
            "match_score": match_score,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
            "skill_gap_analysis": skill_gaps,
            "improvement_suggestions": improvement_suggestions
        }
        
        # Cache response
        background_tasks.add_task(lambda: response_cache.update({cache_key: result}))
        
        return result
    
    except Exception as e:
        logger.error(f"Error comparing resume to job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare resume to job: {str(e)}"
        )

@router.post("/skill-gap/", status_code=status.HTTP_200_OK)
async def analyze_skill_gap(request: ResumeJobRequest):
    """
    Analyze skill gaps between resume and job description
    
    - **resume_text**: Resume text to analyze
    - **job_text**: Job description text to compare against
    
    Returns detailed skill gap analysis
    """
    try:
        # Extract just the skills from both texts
        resume_tech_skills, resume_soft_skills = extract_skills(request.resume_text)
        job_tech_skills, job_soft_skills = extract_skills(request.job_text)
        
        # Convert to sets for comparison
        resume_tech_set = set(s.lower() for s in resume_tech_skills)
        job_tech_set = set(s.lower() for s in job_tech_skills)
        
        resume_soft_set = set(s.lower() for s in resume_soft_skills)
        job_soft_set = set(s.lower() for s in job_soft_skills)
        
        # Find missing and matching skills
        missing_tech_skills = list(job_tech_set - resume_tech_set)
        missing_soft_skills = list(job_soft_set - resume_soft_set)
        
        matching_tech_skills = list(resume_tech_set.intersection(job_tech_set))
        matching_soft_skills = list(resume_soft_set.intersection(job_soft_set))
        
        # Calculate match percentages
        tech_match_percent = 0
        if len(job_tech_set) > 0:
            tech_match_percent = round(len(matching_tech_skills) / len(job_tech_set) * 100, 2)
            
        soft_match_percent = 0
        if len(job_soft_set) > 0:
            soft_match_percent = round(len(matching_soft_skills) / len(job_soft_set) * 100, 2)
        
        return {
            "technical_skills": {
                "matching": matching_tech_skills,
                "missing": missing_tech_skills,
                "match_percentage": tech_match_percent
            },
            "soft_skills": {
                "matching": matching_soft_skills,
                "missing": missing_soft_skills,
                "match_percentage": soft_match_percent
            },
            "overall_skill_match": round((tech_match_percent * 0.7 + soft_match_percent * 0.3), 2)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing skill gap: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze skill gap: {str(e)}"
        )

@router.post("/keyword-relevance/", status_code=status.HTTP_200_OK)
async def analyze_keyword_relevance(request: ResumeJobRequest):
    """
    Analyze keyword relevance between resume and job description
    
    - **resume_text**: Resume text to analyze
    - **job_text**: Job description text to compare against
    
    Returns analysis of keyword importance and relevance
    """
    try:
        # Extract keywords from both texts using TF-IDF
        resume_analysis = analyze_keywords(request.resume_text)
        job_analysis = analyze_keywords(request.job_text)
        
        # Get keywords and their frequencies
        resume_keywords = resume_analysis.get("keyword_frequency", {})
        job_keywords = job_analysis.get("keyword_frequency", {})
        
        # Analyze relevance of resume keywords to job
        relevant_keywords = {}
        missing_important_keywords = []
        
        # For each job keyword, check if it's in resume and at what frequency
        for keyword, job_freq in job_keywords.items():
            if keyword.lower() in [k.lower() for k in resume_keywords.keys()]:
                # Find the actual case-preserved key
                resume_key = next((k for k in resume_keywords.keys() if k.lower() == keyword.lower()), None)
                resume_freq = resume_keywords[resume_key] if resume_key else 0
                
                # Calculate relevance score (higher job freq = more important)
                relevance = (resume_freq / max(resume_keywords.values(), default=1)) * (job_freq / max(job_keywords.values(), default=1))
                relevant_keywords[keyword] = {
                    "job_frequency": job_freq,
                    "resume_frequency": resume_freq,
                    "relevance_score": round(relevance * 100, 2)
                }
            else:
                # This is an important keyword from the job that's missing in the resume
                if job_freq > 1:  # Only include if it appears multiple times in job desc
                    missing_important_keywords.append({
                        "keyword": keyword,
                        "job_frequency": job_freq,
                        "importance": "high" if job_freq > 3 else "medium"
                    })
        
        return {
            "relevant_keywords": relevant_keywords,
            "missing_important_keywords": missing_important_keywords,
            "keyword_match_assessment": "strong" if len(missing_important_keywords) < 3 else 
                                       "moderate" if len(missing_important_keywords) < 7 else 
                                       "weak"
        }
    
    except Exception as e:
        logger.error(f"Error analyzing keyword relevance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze keyword relevance: {str(e)}"
        )