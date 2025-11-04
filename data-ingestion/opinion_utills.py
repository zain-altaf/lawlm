import re
from typing import Dict, Any
from bs4 import BeautifulSoup


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