# AI Resume Enhancer

## Overview

AI Resume Enhancer is an intelligent API service that helps users analyze, improve, and optimize their resumes using natural language processing and machine learning techniques. The application extracts key information from resumes, provides improvement suggestions, and compares resumes against job descriptions to identify potential gaps.

## Key Features

- **Resume Analysis**: Extract keywords, skills, and important phrases from resumes
- **Resume Enhancement**: Generate suggestions to improve resume content and structure
- **Job Matching**: Compare resumes against job descriptions and calculate match scores
- **PDF Processing**: Extract text from PDF resumes
- **AI-Powered Suggestions**: Generate comprehensive improvement recommendations

## Technology Stack

- FastAPI for API development
- Hugging Face Transformers for NLP models
- NLTK and spaCy for natural language processing
- PyPDF2 for PDF parsing
- TF-IDF vectorization for keyword extraction

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-resume-enhancer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. Download spaCy model (optional but recommended):
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Application

### Development Mode

```bash
python main.py --reload
```

### Production Mode

```bash
python main.py
```

By default, the API will run on `http://0.0.0.0:8000`. You can specify a different host or port:

```bash
python main.py --host 127.0.0.1 --port 5000
```

## API Documentation

Once the server is running, access the interactive API documentation:

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## API Endpoints

### Resume Processing

- `POST /api/v1/resume/upload/`: Upload and process a PDF or text resume
- `POST /api/v1/resume/process-text/`: Process plain text resume content
- `POST /api/v1/resume/analyze-full/`: Comprehensive resume analysis

### Keyword Extraction

- `POST /api/v1/keywords/extract/`: Extract keywords, skills, and phrases from resume text
- `POST /api/v1/keywords/skills/`: Extract only technical and soft skills
- `POST /api/v1/keywords/phrases/`: Extract important phrases

### Resume Improvements

- `POST /api/v1/improvements/suggest/`: Generate improvement suggestions
- `POST /api/v1/improvements/enhanced-suggest/`: Generate comprehensive improvements with keyword context

### Job Matching

- `POST /api/v1/job-matching/compare/`: Compare resume to job description
- `POST /api/v1/job-matching/skill-gap/`: Analyze skill gaps
- `POST /api/v1/job-matching/keyword-relevance/`: Analyze keyword relevance

## Example Use Cases

### 1. Resume Analysis

```python
import requests

api_url = "http://localhost:8000/api/v1/resume/process-text/"
resume_text = """
John Doe
Software Engineer
john.doe@example.com

Experience:
Software Engineer, ABC Inc. (2019-Present)
- Developed RESTful APIs using Python and FastAPI
- Implemented CI/CD pipelines with GitHub Actions

Education:
BS Computer Science, XYZ University (2015-2019)
"""

response = requests.post(
    api_url, 
    json={"text": resume_text}
)
print(response.json())
```

### 2. Job Matching

```python
import requests

api_url = "http://localhost:8000/api/v1/job-matching/compare/"
data = {
    "resume_text": "Your resume text here",
    "job_text": "Job description text here"
}

response = requests.post(api_url, json=data)
print(response.json())
```

## Project Structure

```
ai-resume-enhancer/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── improvements.py
│   │   │   ├── job_matching.py
│   │   │   ├── keywords.py
│   │   │   └── resume.py
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai_models.py
│   │   ├── keyword_extractor.py
│   │   ├── pdf_parser.py
│   │   └── text_processor.py
│   └── __init__.py
├── main.py
└── requirements.txt
```
