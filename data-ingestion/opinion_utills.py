import re
import os
import time
import logging
import requests
from typing import Dict, Any, List
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

def api_request_with_retry(url, headers, max_retries=3, retry_delay=2, request_delay=0.5):
    """Make API request with retry logic and rate limiting.

    Args:
        url: URL to request
        headers: Request headers
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        request_delay: Delay before each request in seconds

    Returns:
        Response JSON or None on failure
    """
    for attempt in range(max_retries):
        try:
            time.sleep(request_delay)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"Request failed after {max_retries} attempts: {e}")
                return None
    return None

def extract_legal_info(text: str) -> Dict[str, Any]:
    """
    Extract legal citations and entities from text using regex patterns.

    Args:
        text: Text to extract legal information from

    Returns:
        Dictionary containing citations and legal entities
    """
    # Initialize result structure
    result = {
        'citations': [],
        'entities': {
            'judges': [],
            'parties': [],
            'courts': [],
            'statutes': []
        }
    }
    
    # Citation patterns
    citation_patterns = [
        # U.S. Reports: e.g., "123 U.S. 456 (1987)"
        r'\b\d+\s+U\.S\.?\s+\d+\s*\(\d{4}\)',
        # Federal Reporter: e.g., "123 F.2d 456 (9th Cir. 1987)"
        r'\b\d+\s+F\.\s*(?:2d|3d)?\s+\d+\s*\([^)]*\d{4}\)',
        # Supreme Court Reporter: e.g., "123 S. Ct. 456 (1987)"
        r'\b\d+\s+S\.\s*Ct\.\s+\d+\s*\(\d{4}\)',
        # State cases: e.g., "123 Cal. App. 2d 456 (1987)"
        r'\b\d+\s+[A-Z][a-z]*\.?\s*(?:App\.?\s*)?(?:\d[a-z]*\s+)?\d+\s*\([^)]*\d{4}\)',
        # Law reviews: e.g., "123 Harv. L. Rev. 456 (1987)"
        r'\b\d+\s+[A-Z][a-z]*\.?\s*L\.?\s*Rev\.?\s+\d+\s*\(\d{4}\)'
    ]
    
    # Extract citations
    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)
    result['citations'] = list(set(citations))  # Remove duplicates
    
    # Judge patterns: "Justice [Name]", "Judge [Name]", "Chief Justice [Name]"
    judge_patterns = [
        r'(?:Justice|Judge|Chief Justice|Associate Justice)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'([A-Z][a-z]+),?\s+(?:J\.|C\.J\.|Associate Justice|Chief Justice)'
    ]
    
    for pattern in judge_patterns:
        matches = re.findall(pattern, text)
        result['entities']['judges'].extend([match.strip() for match in matches if isinstance(match, str)])
    
    # Party patterns: "Plaintiff v. Defendant" format
    party_pattern = r'([A-Z][a-zA-Z\s&,\.]+?)\s+v\.?\s+([A-Z][a-zA-Z\s&,\.]+?)(?:\s|,|\.|\n)'
    party_matches = re.findall(party_pattern, text)
    for plaintiff, defendant in party_matches:
        result['entities']['parties'].extend([plaintiff.strip(), defendant.strip()])
    
    # Court patterns
    court_patterns = [
        r'(Supreme Court of [A-Z][a-zA-Z\s]+)',
        r'(United States Supreme Court)',
        r'([A-Z][a-zA-Z\s]+ Circuit Court of Appeals)',
        r'([A-Z][a-zA-Z\s]+ District Court)',
        r'(Court of Appeals for the [A-Z][a-zA-Z\s]+ Circuit)'
    ]
    
    for pattern in court_patterns:
        matches = re.findall(pattern, text)
        result['entities']['courts'].extend(matches)
    
    # Statute patterns: "42 U.S.C. § 1983", "Title VII", etc.
    statute_patterns = [
        r'\b\d+\s+U\.S\.C\.?\s*§+\s*\d+[a-z]*(?:\([^)]+\))*',
        r'Title\s+[IVX]+(?:\s+of\s+[^,.\n]+)?',
        r'Section\s+\d+[a-z]*(?:\([^)]+\))*'
    ]
    
    for pattern in statute_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        result['entities']['statutes'].extend(matches)
    
    # Clean and deduplicate entities
    for key in result['entities']:
        result['entities'][key] = list(set([item.strip() for item in result['entities'][key] if item.strip()]))
    
    return result


def clean_text(content: str) -> str:
    """
    Strip HTML/XML tags and normalize whitespace.

    Args:
        content: Raw text content that may contain HTML/XML

    Returns:
        Cleaned text with normalized whitespace
    """
    if not content:
        return ''
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text)


def enhanced_text_processing(text: str) -> Dict[str, Any]:
    """
    Enhanced processing that extracts citations and legal entities.

    Args:
        text: Raw text to process

    Returns:
        Dictionary containing cleaned text, citations, entities, and statistics
    """
    if not text:
        return {
            'cleaned_text': '',
            'citations': [],
            'legal_entities': {'judges': [], 'parties': [], 'courts': [], 'statutes': []},
            'text_stats': {'length': 0, 'word_count': 0}
        }

    cleaned = clean_text(text)
    legal_info = extract_legal_info(cleaned)

    return {
        'cleaned_text': cleaned,
        'citations': legal_info['citations'],
        'legal_entities': legal_info['entities'],
        'text_stats': {
            'length': len(cleaned),
            'word_count': len(cleaned.split()),
            'citation_count': len(legal_info['citations'])
        }
    }


def fix_chunk_overlaps(chunks: List[str], min_chunk_size_chars: int) -> List[str]:
    """
    Fix chunk overlaps to ensure they start and end at proper sentence boundaries.

    This addresses the issue where RecursiveCharacterTextSplitter's character-based
    overlap creates fragments like 'Moreover, the plaintiffs' contention...'

    Args:
        chunks: List of text chunks to fix

    Returns:
        List of cleaned chunks with proper sentence boundaries
    """
    if not chunks:
        return chunks
    
    fixed_chunks = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 50:
            continue
        
        # Clean up the beginning: remove leading sentence fragments
        chunk = fix_chunk_start(chunk)
        
        # Clean up the end: ensure complete sentences
        chunk = fix_chunk_end(chunk)
        
        # Only keep chunks that meet quality thresholds
        if chunk and len(chunk.strip()) >= min_chunk_size_chars:
            fixed_chunks.append(chunk)
    
    return fixed_chunks
    

def fix_chunk_start(chunk: str) -> str:
    """
    Fix the start of a chunk to begin at a proper sentence boundary.

    Args:
        chunk: Text chunk to fix

    Returns:
        Fixed chunk starting at a sentence boundary
    """
    if not chunk:
        return chunk
        
    # If it already starts well, keep it
    if starts_at_sentence_boundary(chunk):
        return chunk
        
    # Look for sentence patterns: '. [A-Z]', '? [A-Z]', '! [A-Z]', or start of paragraph
    patterns = [
        r'[.!?]\s+[A-Z]',  # Sentence ending + capital letter
        r'\n\s*[A-Z]',     # Paragraph start
        r'^[A-Z]'          # Already starts with capital (fallback)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, chunk)
        if match:
            # Start from the capital letter
            start_pos = match.end() - 1
            return chunk[start_pos:].strip()
    
    # If no good boundary found, return as-is (better than losing content)
    return chunk


def fix_chunk_end(chunk: str) -> str:
    """
    Fix the end of a chunk to end at a complete sentence.

    Args:
        chunk: Text chunk to fix

    Returns:
        Fixed chunk ending at a complete sentence
    """
    if not chunk:
        return chunk
        
    chunk = chunk.rstrip()
    
    # If already ends at sentence boundary, keep it
    if chunk.endswith(('.', '!', '?')):
        return chunk
    
    # Look for the last sentence-ending punctuation
    sentence_endings = list(re.finditer(r'[.!?]', chunk))
    
    if sentence_endings:
        last_sentence = sentence_endings[-1]
        # Keep text up to and including the sentence ending
        return chunk[:last_sentence.end()].rstrip()
    
    # If no sentence endings found, look for other natural breakpoints
    for punct in [';', ':']:
        last_punct = chunk.rfind(punct)
        if last_punct > len(chunk) * 0.8:  # Only if near the end
            return chunk[:last_punct + 1].rstrip()
    
    # As fallback, return as-is
    return chunk


def starts_at_sentence_boundary(text: str) -> bool:
    """
    Check if text starts at a natural sentence boundary.

    Args:
        text: Text to check

    Returns:
        True if text starts at a good sentence boundary
    """
    if not text:
        return False
        
    # Bad starts (sentence fragments) - check these first!
    if text.startswith(('.', ',', ';', ':')):
        return False
    if text.startswith(('moreover,', 'however,', 'furthermore,', 'additionally,')):
        return False
        
    # Good starts
    first_char = text[0]
    if first_char.isupper():
        return True
    if text.startswith(('(', '[', '"', "'")):
        return True
    if text.startswith(('a ', 'an ', 'the ', 'and ', 'or ', 'but ')):
        return True
        
    return False


def get_qdrant_client() -> QdrantClient:
    """
    Get a Qdrant client instance.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    client = QdrantClient(url=qdrant_url)
    return client
