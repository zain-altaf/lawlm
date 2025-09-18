#!/usr/bin/env python3
"""
Unified Legal Document Processing Pipeline

This module orchestrates the complete pipeline:
1. Data ingestion from CourtListener API
2. Chunking of legal texts with overlap
3. Vector processing with BGE embeddings

Provides a single entry point for the entire processing workflow.
"""

# Standard library imports
import argparse
import json
import logging
import os
import re
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local imports
from config import PipelineConfig, load_config

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_url_for_logging(url: str) -> str:
    """Sanitize URLs to remove sensitive parameters before logging."""
    # Remove common sensitive parameters
    sensitive_params = ['api_key', 'token', 'key', 'password', 'secret']
    parsed = urllib.parse.urlparse(url)

    if parsed.query:
        params = urllib.parse.parse_qs(parsed.query)
        sanitized_params = {}

        for key, values in params.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_params):
                sanitized_params[key] = ['***REDACTED***']
            else:
                sanitized_params[key] = values

        sanitized_query = urllib.parse.urlencode(sanitized_params, doseq=True)
        sanitized_url = urllib.parse.urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, sanitized_query, parsed.fragment
        ))
        return sanitized_url

    return url


def sanitize_exception_message(message: str) -> str:
    """Sanitize exception messages to remove potential API keys."""
    # Pattern to match common API key formats
    patterns = [
        r'(api[_-]?key[=:]\s*)([a-zA-Z0-9\-_]{6,})',  # Reduced minimum length
        r'(token[=:]\s*)([a-zA-Z0-9\-_]{6,})',          # Reduced minimum length
        r'(authorization[=:]\s*token\s+)([a-zA-Z0-9\-_]+)',
        r'(token\s+)([a-zA-Z0-9\-_]{6,})',              # Added pattern for "token abc123"
    ]

    sanitized = str(message)
    for pattern in patterns:
        sanitized = re.sub(pattern, r'\1***REDACTED***', sanitized, flags=re.IGNORECASE)

    return sanitized


class CircuitBreaker:
    """Simple circuit breaker implementation for API calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker state: HALF_OPEN - attempting recovery")
            else:
                raise Exception("Circuit breaker is OPEN - API calls blocked")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker state: CLOSED - recovery successful")

    def _on_failure(self):
        """Track failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker state: OPEN - {self.failure_count} failures detected")

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"

# Handle vector processor import gracefully
try:
    from vector_processor import EnhancedVectorProcessor
    VECTOR_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vector processing not available ({e}). Only chunking will work.")
    EnhancedVectorProcessor = None
    VECTOR_PROCESSOR_AVAILABLE = False

# Global circuit breaker for API calls
api_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


def fetch_with_retry(url: str, headers: Dict[str, str] = None, timeout: int = 30,
                    max_retries: int = 3, delay: int = 5, circuit_breaker: CircuitBreaker = None) -> Optional[Dict[str, Any]]:
    """
    Fetch data from URL with retry logic for 5xx server errors and 429 rate limiting.

    Args:
        url: URL to fetch
        headers: HTTP headers to send
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        delay: Seconds to wait between retries
        circuit_breaker: Optional circuit breaker for protection

    Returns:
        JSON response data or None if all retries failed
    """
    def _make_request():
        return requests.get(url, headers=headers or {}, timeout=timeout)

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Use circuit breaker if provided
            if circuit_breaker:
                response = circuit_breaker.call(_make_request)
            else:
                response = _make_request()
            
            # Success - return the data
            if response.status_code == 200:
                return response.json()

            # Rate limit (429) - respect Retry-After header
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                safe_url = sanitize_url_for_logging(url)
                logger.warning(f"Rate limited (429) for {safe_url} - waiting {retry_after}s before retry (attempt {attempt + 1}/{max_retries})")

                if attempt < max_retries:
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error(f"Rate limited for {safe_url} - max retries exceeded")
                    return None

            # Other client errors (4xx) - don't retry, content doesn't exist
            elif 400 <= response.status_code < 500:
                safe_url = sanitize_url_for_logging(url)
                logger.warning(f"Client error {response.status_code} for {safe_url} - not retrying")
                return None

            # Server errors (5xx) - retry with delay
            elif response.status_code >= 500:
                safe_url = sanitize_url_for_logging(url)
                if attempt < max_retries:
                    logger.warning(f"Server error {response.status_code} for {safe_url} - retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Server error {response.status_code} for {safe_url} - max retries exceeded")
                    return None

        except requests.exceptions.RequestException as e:
            safe_url = sanitize_url_for_logging(url)
            safe_error = sanitize_exception_message(str(e))
            if attempt < max_retries:
                logger.warning(f"Request exception for {safe_url}: {safe_error} - retrying in {delay}s")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Request failed for {safe_url} after {max_retries} retries: {safe_error}")
                return None
    
    return None

class LegalDocumentPipeline:
    """
    Unified pipeline for processing legal documents from ingestion to search-ready vectors.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the legal document processing pipeline.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.working_dir = Path(self.config.processing.working_directory)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🏛️ Legal Document Pipeline initialized")
        logger.info(f"📂 Working directory: {self.working_dir}")
        logger.info(f"🤖 Embedding model: {self.config.vector_processing.embedding_model}")


    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and available data.

        Returns:
            Dict containing working directory, files, and configuration summary
        """
        return {
            'working_dir': str(self.working_dir),
            'files': [str(f) for f in self.working_dir.glob("*")],
            'configuration': self.config.get_summary()
        }


    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the legal document processing pipeline.

        Fetches existing dockets to prevent duplication, processes new dockets only,
        and includes smart pagination for efficient API calls.

        Returns:
            Dict containing pipeline execution results and statistics
        """
        logger.info(f"🚀 Starting legal document processing pipeline")
        logger.info(f"🏛️ Court: {self.config.data_ingestion.court}, Dockets: {self.config.data_ingestion.num_dockets}")
        start_time = datetime.now()
        
        # Initialize vector processor
        vector_processor = None
        if VECTOR_PROCESSOR_AVAILABLE and self.config.vector_processing.collection_name_vector:
            vector_processor = EnhancedVectorProcessor(
                model_name=self.config.vector_processing.embedding_model,
                collection_name=self.config.vector_processing.collection_name_vector,
                qdrant_url=self.config.qdrant.url
            )
            logger.info(f"🤖 Vector processor initialized with deduplication")

        # Initialize text splitter with enhanced separators for legal text
        # Prioritize paragraphs, then sentences, then lines, then words
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.text_splitter.chunk_size_chars,
            chunk_overlap=self.config.text_splitter.overlap_chars,
            length_function=len,
            separators=[
                "\n\n",    # Paragraph breaks (highest priority)
                ". ",      # Sentence endings  
                "? ",      # Question endings
                "! ",      # Exclamation endings
                "\n",      # Line breaks
                " ",       # Word boundaries  
                ""         # Character level (last resort)
            ]
        )
        
        # Get all existing dockets in Qdrant
        existing_dockets = set()
        if vector_processor:
            existing_dockets = vector_processor.get_existing_docket_ids()
            logger.info(f"🔍 Found {len(existing_dockets)} existing dockets in collection")

        # Set starting ID for new documents
        qdrant_id = len(existing_dockets) + 1

        # Stats tracking
        stats = {
            'total_requested': self.config.data_ingestion.num_dockets,
            'dockets_fetched': 0,
            'dockets_processed': 0,
            'opinions_processed': 0,
            'chunks_created': 0,
            'vectors_uploaded': 0,
            'duplicates_skipped': 0,
            'errors': []
        }
        
        try:
            # Step 1: Smart pagination that fetches NEW dockets directly (deduplication built-in)
            logger.info(f"📋 Smart fetching dockets with built-in deduplication...")
            new_dockets = self._fetch_all_dockets_paginated(
                self.config.data_ingestion.court,
                self.config.data_ingestion.num_dockets,
                existing_dockets
            )

            stats['dockets_fetched'] = len(new_dockets)
            stats['duplicates_skipped'] = 0
            
            logger.info(f"📊 Smart pagination found {len(new_dockets)} new dockets to process")
            
            if not new_dockets:
                logger.info("✅ No new dockets to process - collection is up to date")
                return {
                    'status': 'up_to_date',
                    'stats': stats,
                    'message': 'All requested dockets already exist in collection'
                }
            
            # Step 2: Process each docket incrementally  
            for idx, docket in enumerate(new_dockets, 1):
                docket_id = docket.get('id', '')
                if not docket_id:
                    continue
                    
                logger.info(f"\n[{idx}/{len(new_dockets)}] Processing docket {docket_id}")
                
                try:
                    # Fetch all opinions for all cluster in this specific docket directly
                    _, docket_opinions = self._fetch_docket_clusters_and_opinions(docket, self.config.data_ingestion.court, vector_processor)

                    stats['opinions_processed'] += len(docket_opinions)
                    
                    if not docket_opinions:
                        logger.warning(f"⚠️ No opinions found for docket {docket_id}, skipping")
                        continue
                    
                    # Chunk documents
                    all_chunks = []
                    for op in docket_opinions:
                        opinion_text = op.get('opinion_text', '')
                        if not opinion_text or len(opinion_text.strip()) < 50:
                            continue

                        raw_chunks = text_splitter.split_text(opinion_text)
                        
                        # Clean up chunk boundaries and fix overlaps to ensure complete sentences
                        text_chunks = self._fix_chunk_overlaps(raw_chunks)
                        
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            if len(chunk_text.strip()) < self.config.text_splitter.min_chunk_size_chars:
                                continue

                            chunk = {
                                "id": qdrant_id,
                                "docket_id": docket_id,
                                "cluster_id": op.get('cluster_id', ''),
                                "opinion_id": op.get('opinion_id', ''),
                                "chunk_id": f"{op.get('opinion_id', 'unknown')}_{chunk_idx}",
                                "docket_number": docket.get('docket_number', ''),
                                "case_name": op.get('case_name', ''),
                                "court_id": op.get('court_id', ''),
                                "chunk_index": chunk_idx,
                                "judges": op.get('judges', ''),
                                "author": op.get('author', ''),
                                "opinion_type": op.get('opinion_type', ''),
                                "date_filed": op.get('date_filed', ''),
                                "text": chunk_text.strip(),
                                "precedential_status": op.get('precedential_status', ''),
                                "sha1": op.get('sha1', ''),
                                "download_url": op.get('download_url', ''),
                                "source_field": op.get('source_field', ''),
                                "citations": op.get('citations', []),
                                "legal_entities": op.get('legal_entities', {}),
                                "date_created": op.get('date_created', ''),
                                "date_modified": op.get('date_modified', '')
                            }

                            all_chunks.append(chunk)
                                            
                    stats['chunks_created'] += len(all_chunks)
                    
                    # Vectorize and upload immediately
                    if vector_processor and all_chunks:
                        logger.info(f"🔄 Vectorizing and uploading {len(all_chunks)} chunks")
                        
                        # Process in small batches
                        batch_size = min(self.config.vector_processing.batch_size, 50)
                        for i in range(0, len(all_chunks), batch_size):
                            batch = all_chunks[i:i + batch_size]
                            try:
                                result = vector_processor.process_and_upload_chunks(
                                    batch,
                                    collection_name=self.config.vector_processing.collection_name_vector
                                )
                                stats['vectors_uploaded'] += result.get('vectors_created', 0)
                            except Exception as e:
                                logger.error(f"Failed to upload batch: {e}")
                                stats['errors'].append({'docket': docket_id, 'error': str(e)})
                    
                    stats['dockets_processed'] += 1
                    logger.info(f"✅ Docket {docket_id}: {len(docket_opinions)} opinions, {len(all_chunks)} chunks")

                except Exception as e:
                    logger.error(f"❌ Error processing docket {docket_id}: {e}")
                    stats['errors'].append({'docket': docket_id, 'error': str(e)})

                qdrant_id += 1
            
            # Final summary
            duration = (datetime.now() - start_time).total_seconds()
            
            final_summary = {
                'status': 'completed',
                'pipeline': 'standard',
                'court': self.config.data_ingestion.court,
                'duration_seconds': duration,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save summary
            summary_file = self.working_dir / "pipeline_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2)
            
            logger.info(f"\n{'='*50}")
            logger.info(f"✅ Pipeline completed!")
            logger.info(f"📊 Requested: {stats['total_requested']} dockets")
            logger.info(f"📊 Fetched: {stats['dockets_fetched']} dockets")
            logger.info(f"📊 Processed: {stats['dockets_processed']} new dockets")
            logger.info(f"📊 Skipped duplicates: {stats['duplicates_skipped']}")
            logger.info(f"📄 Opinions: {stats['opinions_processed']}")
            logger.info(f"🔪 Chunks: {stats['chunks_created']}")
            logger.info(f"🔮 Vectors: {stats['vectors_uploaded']}")
            logger.info(f"⏱️ Duration: {duration:.2f} seconds")
            if stats['errors']:
                logger.warning(f"⚠️ Errors: {len(stats['errors'])}")
            logger.info(f"💾 Summary saved to: {summary_file}")
            
            return final_summary
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


    def _fetch_all_dockets_paginated(self, court: str, num_dockets: int, existing_dockets: set) -> List[Dict[str, Any]]:
        """
        Fetch dockets using cursor-based pagination starting from oldest first.

        This ensures consistent ordering and prevents pagination issues with new dockets.

        Args:
            court: Court identifier (e.g., 'scotus')
            num_dockets: Number of new dockets to fetch
            existing_dockets: Set of existing docket IDs to avoid duplicates

        Returns:
            List of new docket dictionaries
        """

        load_dotenv()
        api_key = os.getenv('CASELAW_API_KEY')
        headers = {'Authorization': f'Token {api_key}'} if api_key else {}
        
        new_dockets = []
        page_count = 0
        page_size = 200
        consecutive_empty_pages = 0
        max_consecutive_empty = 50
        
        logger.info(f"🌐 Fetching {num_dockets} new dockets from {court}")
        logger.info(f"🔍 Found {len(existing_dockets)} existing dockets in collection")

        base_url = "https://www.courtlistener.com/api/rest/v4/dockets/"

        # Start with cursor-based pagination from the beginning (oldest first)
        cursor = None
        
        while len(new_dockets) < num_dockets:
            page_count += 1
            
            logger.info(f"📄 Fetching page {page_count} - need {num_dockets - len(new_dockets)} more NEW dockets")
            
            try:
                params = {
                    "court": court,
                    "ordering": "id"  # Oldest dockets first for consistent pagination
                }
                if cursor:
                    params["cursor"] = cursor
                
                # Construct full URL with params for retry function
                query_string = urllib.parse.urlencode(params)
                full_url = f"{base_url}?{query_string}"
                
                response_data = fetch_with_retry(
                    url=full_url,
                    headers=headers,
                    timeout=30,
                    max_retries=3,
                    delay=5,
                    circuit_breaker=api_circuit_breaker
                )
                
                if response_data is None:
                    logger.error(f"Failed to fetch dockets page {page_count} after retries")
                    break
                page_dockets = response_data.get('results', [])
                if not page_dockets:
                    logger.warning(f"⚠️ No more dockets available from API")
                    break
                
                # Get next cursor for continuation
                next_url = response_data.get('next')
                if next_url:
                    parsed = urllib.parse.urlparse(next_url)
                    query_params = urllib.parse.parse_qs(parsed.query)
                    cursor = query_params.get('cursor', [None])[0]
                else:
                    cursor = None
                
                # Check this page for new dockets
                page_new_count = 0
                for docket in page_dockets:
                    docket_id = docket.get('id', '')
                    if docket_id and docket_id not in existing_dockets:
                        new_dockets.append(docket)
                        page_new_count += 1

                        # Stop if we have enough new dockets
                        if len(new_dockets) >= num_dockets:
                            break

                logger.info(f"📊 Page {page_count}: {len(page_dockets)} total, {page_new_count} new (total new: {len(new_dockets)})")
                
                # Track consecutive empty pages
                if page_new_count == 0:
                    consecutive_empty_pages += 1
                    logger.info(f"⚠️ Page {page_count} had no new dockets ({consecutive_empty_pages}/{max_consecutive_empty})")
                    
                    if consecutive_empty_pages >= max_consecutive_empty:
                        logger.warning(f"🛑 Stopping: {max_consecutive_empty} consecutive pages with no new dockets")
                        break
                else:
                    consecutive_empty_pages = 0  # Reset counter
                
                # Check if we have enough new dockets
                if len(new_dockets) >= num_dockets:
                    logger.info(f"✅ Found enough new dockets: {len(new_dockets)}")
                    break
                
                # Check if we've reached the end of available data
                if not cursor:
                    logger.info(f"📋 Reached end of available dockets (no more pages)")
                    break
                
            except Exception as e:
                logger.error(f"❌ Unexpected error on page {page_count}: {e}")
                break
        
        final_new_dockets = new_dockets[:num_dockets]  # Limit to exactly what was requested
        
        logger.info(f"🎯 Cursor-based pagination complete:")
        logger.info(f"   📄 Pages fetched: {page_count}")
        logger.info(f"   ✨ New dockets found: {len(final_new_dockets)}")
        logger.info(f"   📊 Ordering: oldest first (id) for consistent deduplication")
        
        return final_new_dockets


    def _fix_chunk_overlaps(self, chunks: List[str]) -> List[str]:
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
            chunk = self._fix_chunk_start(chunk)
            
            # Clean up the end: ensure complete sentences
            chunk = self._fix_chunk_end(chunk)
            
            # Only keep chunks that meet quality thresholds
            if chunk and len(chunk.strip()) >= self.config.text_splitter.min_chunk_size_chars:
                fixed_chunks.append(chunk)
        
        return fixed_chunks
    

    def _fix_chunk_start(self, chunk: str) -> str:
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
        if self._starts_at_sentence_boundary(chunk):
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
    

    def _fix_chunk_end(self, chunk: str) -> str:
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
    

    def _starts_at_sentence_boundary(self, text: str) -> bool:
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
    

    def _fetch_docket_clusters_and_opinions(self, docket: Dict[str, Any], court: str, vector_processor=None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch and process all clusters and opinions for a specific docket object.

        Args:
            docket: Docket dictionary from API
            court: Court identifier
            vector_processor: Optional vector processor for text enhancement

        Returns:
            Tuple of (clusters, opinions) lists
        """

        load_dotenv()
        api_key = os.getenv('CASELAW_API_KEY')
        headers = {'Authorization': f'Token {api_key}'} if api_key else {}
        
        opinions = []
        clusters = []
        docket_id = docket.get('id', '')
        
        logger.info(f"📥 Fetching clusters and opinions for docket {docket_id}")

        # Process clusters and opinions from the docket
        for cluster_url in docket.get("clusters", []):
            try:
                logger.debug(f"Processing cluster: {cluster_url}")
                cluster = fetch_with_retry(
                    url=cluster_url,
                    headers=headers,
                    timeout=30,
                    max_retries=3,
                    delay=5,
                    circuit_breaker=api_circuit_breaker
                )
                if cluster is None:
                    logger.warning(f"Failed to fetch cluster {cluster_url} after retries")
                    continue
                clusters.append(cluster)
                cluster_id = cluster.get('id', '')

                for opinion_url in cluster.get("sub_opinions", []):
                    try:
                        logger.debug(f"Processing opinion: {opinion_url}")
                        opinion = fetch_with_retry(
                            url=opinion_url,
                            headers=headers,
                            timeout=30,
                            max_retries=3,
                            delay=5,
                            circuit_breaker=api_circuit_breaker
                        )
                        if opinion is None:
                            logger.warning(f"Failed to fetch opinion {opinion_url} after retries")
                            continue
                        
                        # Extract text from available fields in order of recommendation
                        raw_text = None
                        source_field = None
                        for field in [
                            'html_with_citations',
                            'plain_text',
                            'html_columbia',
                            'html_lawbox',
                            'html_anon_2020',
                            'html'
                        ]:
                            if opinion.get(field):
                                raw_text = opinion[field]
                                source_field = field
                                break
                        
                        if not raw_text or len(raw_text.strip()) < 100:
                            logger.debug(f"Skipping opinion {opinion.get('id')} - insufficient text")
                            continue
                        
                        # Process text using existing enhanced processing
                        try:
                            processed = vector_processor.enhanced_text_processing(raw_text)
                            
                            opinion = {
                                "docket_id": docket_id,
                                "cluster_id": cluster_id,
                                "opinion_id": opinion.get("id"),
                                "case_name": cluster.get("case_name", "Unknown Case"),
                                "court_id": court,
                                "judges": cluster.get("judges", ""),
                                "author": opinion.get("author_id", ""),
                                "opinion_type": opinion.get("type", "unknown"),
                                "date_filed": cluster.get("date_filed"),
                                "precedential_status": cluster.get("precedential_status"),
                                "sha1": opinion.get("sha1"),
                                "download_url": opinion.get("download_url"),
                                "source_field": source_field,
                                "opinion_text": processed['cleaned_text'],
                                "citations": processed['citations'],
                                "legal_entities": processed['legal_entities'],
                                "text_stats": processed['text_stats'],
                                "date_created": opinion.get("date_created"),
                                "date_modified": opinion.get("date_modified"),
                                "ready_for_chunking": True
                            }
                            
                            opinions.append(opinion)
                            logger.debug(f"Successfully processed opinion {opinion.get('id')}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to process text for opinion {opinion.get('id')}: {e}")
                            continue
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch opinion {opinion_url}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to fetch cluster {cluster_url}: {e}")
                continue

        logger.info(f"📄 Found {len(opinions)} opinions and {len(clusters)} clusters in docket {docket_id}")
        return clusters, opinions

def main() -> None:
    """
    Main entry point for the legal document processing pipeline.

    Parses command line arguments, loads configuration, and runs the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Unified Legal Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', help='Configuration file path (JSON)')
    parser.add_argument('--status', action='store_true', help='Show pipeline status and exit')
    parser.add_argument('--court', help='Court identifier (e.g., scotus, ca1)')
    parser.add_argument('--num-dockets', type=int, help='Number of dockets to process')

    args = parser.parse_args()

    # Load config from JSON if path provided, otherwise load_config()
    # from config.py as default
    config = load_config(args.config) if args.config else load_config()
    
    # Override config with command line arguments
    if args.court:
        config.data_ingestion.court = args.court
    if args.num_dockets:
        config.data_ingestion.num_dockets = args.num_dockets

    # Initialize pipeline
    pipeline = LegalDocumentPipeline(config=config)

    if args.status:
        status = pipeline.get_pipeline_status()

        # Add configuration diagnostics to status output
        try:
            from config import validate_configuration_sources, print_configuration_guidance

            print("="*60)
            print("📊 PIPELINE STATUS")
            print("="*60)
            print(json.dumps(status, indent=2))

            print("\n" + "="*60)
            print("🔧 CONFIGURATION DIAGNOSTICS")
            print("="*60)

            # Run configuration validation and print guidance
            print_configuration_guidance()

        except ImportError:
            print("Configuration diagnostics not available")
            print(json.dumps(status, indent=2))
        return

    # Run pipeline
    results = pipeline.run_pipeline()
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()