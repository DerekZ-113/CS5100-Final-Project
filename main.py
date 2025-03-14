from fastapi import FastAPI, UploadFile, File
from typing import List
from pydantic import BaseModel
import uvicorn     #  Runs the FastAPI application
import re
from transformers import pipeline
import PyPDF2      # Allows extracting text from PDFs
import io
from collections import Counter

# Initialize FastAPI
app = FastAPI()

# Load Hugging Face NLP model for keyword extraction
keyword_extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Define ResumeAnalysisRequest model
class ResumeAnalysisRequest(BaseModel):
  text: str

# Define ResumeJobRequest model
class ResumeJobRequest(BaseModel):
  resume_text: str
  job_text: str

# Root endpoint
@app.get("/")
async def root():
  return {"message": "AI Resume Enhancer Backend is Running!"}

# Upload Resume Endpoint
# Accepts a resume (PDF or TXT) and reads it (extracts text)
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
  try:
      if file.content_type == "application/pdf":
          pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
          text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
      elif file.content_type == "text/plain":
          text = (await file.read()).decode("utf-8")
      else:
          return {"error": "Unsupported file type. Please upload a PDF or TXT file."}
    
      return {"filename": file.filename, "extracted_text": text[:500] + "..."}  # Limit preview text
  except Exception as e:
      return {"error": f"Failed to process resume: {str(e)}"}

# Analyze Keywords in Resume
# Extracts important keywords using Hugging Face
# Keeps keywords with a confidence score greater than 85%
@app.post("/analyze_keywords/")
async def analyze_keywords(request: ResumeAnalysisRequest):
  text = request.text
  keywords = keyword_extractor(text)
  extracted_keywords = [keyword["word"] for keyword in keywords if keyword["score"] > 0.85]  # Filtering high-confidence keywords

  return {"keywords": extracted_keywords}

# Suggest Resume Improvements
# Uses an AI model to suggest resume phrasing improvements
@app.post("/suggest_improvements/")
async def suggest_improvements(request: ResumeAnalysisRequest):
  text = request.text
  improved_text = text.replace("Responsible for", "Successfully led")  # Example improvement
  
  return {"original_text": text[:500] + "...", "improved_text": improved_text[:500] + "..."}

# Compare resume to job posting
# Matches resume keywords with job posting keywords
@app.post("/compare_resume_to_job/")
async def compare_resume_to_job(request: ResumeJobRequest):
  def extract_keywords(text, min_score=0.85):
      extracted = keyword_extractor(text)
      return set(keyword["word"] for keyword in extracted if keyword["score"] > min_score)

  resume_keywords = extract_keywords(request.resume_text)
  job_keywords = extract_keywords(request.job_text)

  matched_keywords = resume_keywords.intersection(job_keywords)
  missing_keywords = job_keywords - resume_keywords
  match_score = round(len(matched_keywords) / max(len(job_keywords), 1) * 100, 2)

  return {
      "match_score": match_score,
      "matched_keywords": list(matched_keywords),
      "missing_keywords": list(missing_keywords)
  }

# Runs API using Uvicorn
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)