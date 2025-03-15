"""
PDF Parser Service

This module handles parsing and text extraction from PDF resumes and other documents.
It provides functions to extract text from PDFs with fallback mechanisms for handling 
various PDF formats and potential extraction issues.
"""

import io
import logging
import re
import PyPDF2
from typing import Optional, Dict, Any

# Set up logger
logger = logging.getLogger("ai-resume-enhancer")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text content from PDF bytes with enhanced error handling
    
    Args:
        pdf_bytes: Raw bytes of the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        ValueError: If text extraction fails
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        
        # Try standard text extraction first
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        # If standard extraction fails, try fallback method
        if not text.strip():
            text = extract_text_fallback(pdf_reader)
        
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF")
            
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Could not extract text from PDF: {str(e)}. The file may be corrupted, password-protected, or contain only images.")

def extract_text_fallback(pdf_reader: Any) -> str:
    """
    Fallback method for extracting text from PDF pages when standard extraction fails
    
    Args:
        pdf_reader: PyPDF2 PdfReader object
        
    Returns:
        Extracted text as a string
    """
    # This is a placeholder for a more robust extraction method
    # In a production implementation, you might use other libraries like pdfminer.six or PyMuPDF
    text = ""
    
    try:
        for page in pdf_reader.pages:
            # Try to extract text from form fields if present
            if hasattr(page, '/Annots') and page['/Annots']:
                for annot in page['/Annots']:
                    obj = annot.get_object()
                    if '/T' in obj and '/V' in obj:
                        text += f"{obj['/T']}: {obj['/V']}\n"
            
            # Try direct string extraction (less reliable)
            page_content = page.get("/Contents", None)
            if page_content:
                content_stream = page_content.get_data() if hasattr(page_content, 'get_data') else ""
                text += str(content_stream) + "\n"
    except Exception as e:
        logger.warning(f"Fallback text extraction encountered an error: {str(e)}")
    
    return text

def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize text extracted from PDF
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Remove duplicate spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def extract_metadata_from_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Extract metadata from PDF document
    
    Args:
        pdf_bytes: Raw bytes of the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    metadata = {
        'title': None,
        'author': None,
        'creator': None,
        'producer': None,
        'creation_date': None,
        'modification_date': None,
        'page_count': 0,
    }
    
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        # Extract basic metadata
        if pdf_reader.metadata:
            for key in metadata:
                pdf_key = f'/{key.capitalize()}' if key != 'page_count' else None
                if pdf_key and pdf_key in pdf_reader.metadata:
                    metadata[key] = pdf_reader.metadata[pdf_key]
        
        # Add page count
        metadata['page_count'] = len(pdf_reader.pages)
        
        return metadata
    except Exception as e:
        logger.warning(f"Could not extract PDF metadata: {str(e)}")
        return metadata

def process_resume_file(file_content: bytes, file_type: str) -> Dict[str, Any]:
    """
    Process a resume file (PDF or TXT) and extract text and metadata
    
    Args:
        file_content: Raw file content as bytes
        file_type: MIME type of the file
        
    Returns:
        Dictionary with extracted text and metadata
        
    Raises:
        ValueError: If file processing fails
    """
    result = {
        'extracted_text': '',
        'metadata': {},
        'file_type': file_type,
        'file_size': len(file_content),
    }
    
    try:
        if file_type == "application/pdf":
            result['extracted_text'] = extract_text_from_pdf(file_content)
            result['metadata'] = extract_metadata_from_pdf(file_content)
        elif file_type == "text/plain":
            result['extracted_text'] = file_content.decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Clean the extracted text
        result['extracted_text'] = clean_extracted_text(result['extracted_text'])
        
        # Create a preview
        preview_length = min(500, len(result['extracted_text']))
        result['text_preview'] = result['extracted_text'][:preview_length] + \
                               ("..." if len(result['extracted_text']) > preview_length else "")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise ValueError(f"Failed to process file: {str(e)}")