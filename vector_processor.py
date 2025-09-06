"""
Enhanced Vector Processor for semantic chunks with improved embeddings.

This processor handles the creation of dense vector embeddings using
BAAI/bge-small-en-v1.5 for better legal domain performance. Optimized
for semantically chunked legal documents.
"""

import uuid
import os
import json
import logging
import gc
from typing import List, Dict, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import re
from bs4 import BeautifulSoup

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available ({e}). CPU-only processing will be used.")
    torch = None
    TORCH_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


def extract_legal_info(text: str) -> Dict[str, Any]:
    """Extract legal citations and entities from text using regex patterns."""
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
    """Strips HTML/XML tags and normalizes whitespace."""
    if not content:
        return ''
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text)


def enhanced_text_processing(text: str) -> Dict[str, Any]:
    """Enhanced processing that extracts citations and legal entities."""
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


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    usage = {
        'ram_mb': memory_info.rss / 1024 / 1024,
        'ram_percent': process.memory_percent()
    }
    
    
    return usage


class EnhancedVectorProcessor:
    """Enhanced processor for semantic chunks using improved embeddings."""
    
    def __init__(self, 
                 model_name: str = 'BAAI/bge-small-en-v1.5',
                 collection_name: str = "caselaw-chunks-vector",
                 qdrant_url: str = None):
        """
        Initialize the enhanced vector processor.
        
        Args:
            model_name: Name of the sentence transformer model (default: BAAI/bge-small-en-v1.5)
            collection_name: Name of the Qdrant collection
            qdrant_url: URL of the Qdrant server
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        
        # Model-specific vector sizes
        model_sizes = {
            'BAAI/bge-small-en-v1.5': 384,
            'all-MiniLM-L6-v2': 384,
            'BAAI/bge-base-en-v1.5': 768,
            'BAAI/bge-large-en-v1.5': 1024
        }
        self.vector_size = model_sizes.get(model_name, 384)
        
        # Load the embedding model with optimization for legal text
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        
        # Optimize model for inference
        self.embedder.eval()
        logger.info("Using CPU for embeddings")
        
        # BGE models work better with specific prompts for retrieval
        if 'bge' in model_name.lower():
            self.query_prefix = "Represent this query for searching relevant legal passages: "
            self.passage_prefix = "Represent this legal passage: "
        else:
            self.query_prefix = ""
            self.passage_prefix = ""
        
        logger.info(f"Vector size: {self.vector_size}")
        logger.info("Enhanced embedding model loaded successfully")
        
        # Initialize Qdrant client
        self.client = self._get_qdrant_client()
    
    
    def _get_qdrant_client(self) -> QdrantClient:
        """Initialize and return Qdrant client with cloud support."""
        try:
            # Check if we have API key for cloud authentication
            api_key = os.getenv("QDRANT_API_KEY")
            
            if api_key:
                # Use API key for cloud authentication
                client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=api_key,
                    timeout=30
                )
                logger.info(f"Connected to Qdrant Cloud at {self.qdrant_url}")
            else:
                # Local instance without API key
                client = QdrantClient(self.qdrant_url)
                logger.info(f"Connected to local Qdrant at {self.qdrant_url}")
            
            # Test connection
            client.get_collections()
            return client
            
        except Exception as e:
            logger.error(f"Error connecting to Qdrant at {self.qdrant_url}: {e}")
            if "Unauthorized" in str(e) or "401" in str(e):
                logger.error("Authentication failed. Check your QDRANT_API_KEY environment variable.")
            elif "cloud.qdrant.io" in self.qdrant_url and not os.getenv("QDRANT_API_KEY"):
                logger.error("Cloud URL detected but no QDRANT_API_KEY provided. Set your API key.")
            raise
    

    def _create_enhanced_text(self, chunk: Dict[str, Any]) -> str:
        """Create enhanced text representation for better embeddings with full legal extraction."""
        text = chunk.get('text', '')
        
        extracted_info = enhanced_text_processing(text)
        citations = extracted_info.get('citations', [])
        entities = extracted_info.get('legal_entities', {})
        
        case_name = chunk.get('case_name', '')
        semantic_topic = chunk.get('semantic_topic', '')
        author = chunk.get('author', '')
        
        enhanced_parts = []
        
        if case_name:
            enhanced_parts.append(f"Case: {case_name}")
        
        if semantic_topic and semantic_topic != 'general':
            enhanced_parts.append(f"Topic: {semantic_topic}")
        
        if author:
            enhanced_parts.append(f"Author: {author}")
        
        if citations:
            enhanced_parts.append(f"Citations: {', '.join(citations[:5])}")  # Limit to first 5
        
        if entities.get('judges'):
            enhanced_parts.append(f"Judges: {', '.join(entities['judges'][:3])}")  # Limit to first 3
        
        if entities.get('statutes'):
            enhanced_parts.append(f"Statutes: {', '.join(entities['statutes'][:3])}")  # Limit to first 3
        
        enhanced_parts.append(extracted_info['cleaned_text'])
        
        enhanced_text = " | ".join(enhanced_parts)
        
        # Add BGE passage prefix if applicable
        if self.passage_prefix:
            enhanced_text = self.passage_prefix + enhanced_text
        
        chunk['extracted_citations'] = citations
        chunk['extracted_entities'] = entities
        
        return enhanced_text
    

    def get_existing_document_ids(self) -> set:
        """Get set of existing document IDs in the collection."""
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                return set()
            
            existing_ids = set()
            offset = None
            
            while True:
                # Scroll through existing points to get document IDs
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Only need payload
                )
                
                if not points:
                    break
                
                for point in points:
                    doc_id = point.payload.get('document_id')
                    if doc_id:
                        existing_ids.add(doc_id)
                
                offset = next_offset
                if next_offset is None:
                    break
            
            return existing_ids
            
        except Exception as e:
            logger.warning(f"Could not get existing document IDs: {e}")
            return set()
    

    def get_existing_docket_numbers(self) -> set:
        """Get set of existing docket numbers in the collection."""
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                return set()
            
            existing_dockets = set()
            offset = None
            
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                
                for point in points:
                    docket = point.payload.get('docket_number')
                    if docket:
                        existing_dockets.add(docket)
                
                offset = next_offset
                if next_offset is None:
                    break
            
            return existing_dockets
            
        except Exception as e:
            logger.warning(f"Could not get existing docket numbers: {e}")
            return set()
    

    def filter_duplicate_chunks(self, chunks: List[Dict[str, Any]], 
                               check_mode: str = "document_id") -> tuple[List[Dict[str, Any]], int, int]:
        """
        Filter out chunks that already exist in the collection.
        
        Args:
            chunks: List of chunks to process
            check_mode: "document_id", "docket_number", or "both"
            
        Returns:
            (filtered_chunks, duplicates_found, total_chunks)
        """
        total_chunks = len(chunks)
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"📋 Collection '{self.collection_name}' doesn't exist - all chunks are new")
            return chunks, 0, total_chunks
        
        logger.info(f"🔍 Checking for duplicates in existing collection...")
        
        if check_mode == "document_id":
            existing_ids = self.get_existing_document_ids()
            filtered_chunks = [chunk for chunk in chunks 
                             if chunk.get('document_id') not in existing_ids]
            
        elif check_mode == "docket_number":
            existing_dockets = self.get_existing_docket_numbers()
            filtered_chunks = [chunk for chunk in chunks 
                             if chunk.get('docket_number') not in existing_dockets]
            
        elif check_mode == "both":
            existing_ids = set(self.get_existing_document_ids())
            existing_dockets = set(self.get_existing_docket_numbers())

            filtered_chunks = []
            for chunk in chunks:
                docket = chunk.get('docket_number')
                doc_id = chunk.get('document_id')

                # Prioritize docket_number
                if docket and docket in existing_dockets:
                    continue  # duplicate → skip
                if doc_id and doc_id in existing_ids:
                    continue  # duplicate by doc_id → skip
                filtered_chunks.append(chunk)

        else:
            raise ValueError(f"Invalid check_mode: {check_mode}")
        
        duplicates_found = total_chunks - len(filtered_chunks)
        
        logger.info(f"📊 Duplicate analysis:")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   New chunks: {len(filtered_chunks)}")
        logger.info(f"   Duplicates found: {duplicates_found}")
        
        return filtered_chunks, duplicates_found, total_chunks
    

    def get_collection_size_mb(self) -> float:
        """Get approximate size of collection in MB."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            # Rough estimate: each vector (384 dimensions) + metadata ≈ 2-3KB
            estimated_size_mb = collection_info.points_count * 0.002  # 2KB per point
            return estimated_size_mb
        except Exception as e:
            logger.warning(f"Could not get collection size: {e}")
            return 0.0

    
    def process_and_upload_batch(self, chunks: List[Dict[str, Any]], 
                                collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process and immediately upload a batch of chunks to Qdrant.
        Designed for incremental processing where each batch is uploaded immediately.
        
        Args:
            chunks: List of chunks to process and upload
            collection_name: Optional collection name (uses self.collection_name if not provided)
            
        Returns:
            Dict with processing results
        """
        if not chunks:
            return {'status': 'no_chunks', 'vectors_created': 0}
        
        collection = collection_name or self.collection_name
        
        try:
            # Ensure collection exists
            if not self.client.collection_exists(collection_name=collection):
                # Create collection with proper vector configuration
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
            
            # Create enhanced texts and embeddings for chunks
            enhanced_texts = []
            for chunk in chunks:
                enhanced_text = self._create_enhanced_text(chunk)
                enhanced_texts.append(enhanced_text)
            
            embeddings = self.embedder.encode(
                enhanced_texts,
                batch_size=min(len(enhanced_texts), 16),
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Prepare points for upload
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                
                # Create payload with all chunk metadata including extracted entities
                payload = {
                    'chunk_id': chunk.get('chunk_id', point_id),
                    'document_id': chunk.get('document_id', ''),
                    'docket_number': chunk.get('docket_number', ''),
                    'case_name': chunk.get('case_name', ''),
                    'court_id': chunk.get('court_id', ''),
                    'chunk_index': chunk.get('chunk_index', i),
                    'text': chunk['text'],
                    'author': chunk.get('author', ''),
                    'opinion_type': chunk.get('opinion_type', ''),
                    'date_filed': chunk.get('date_filed', ''),
                    'token_count': chunk.get('token_count', len(chunk['text'].split())),
                    'citation_count': len(chunk.get('extracted_citations', [])),
                    'citations': chunk.get('extracted_citations', []),
                    'legal_entities': chunk.get('extracted_entities', {})
                }
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=collection,
                points=points
            )
            
            return {
                'status': 'success',
                'vectors_created': len(points),
                'collection': collection
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return {
                'status': 'error',
                'vectors_created': 0,
                'error': str(e)
            }
    
    
    def process_chunks(self, chunks: List[Dict[str, Any]], 
                      batch_size: int = 50, 
                      checkpoint_interval: int = 100,
                      skip_duplicates: bool = True,
                      duplicate_check_mode: str = "document_id",
                      overwrite_collection: bool = False) -> str:
        """
        Process semantic chunks for enhanced vector search with duplicate detection.
        
        Args:
            chunks: List of semantic chunk dictionaries from smart_chunking
            batch_size: Number of chunks to process in each batch (default: 50)
            checkpoint_interval: Interval for progress logging and cleanup (default: 100)
            skip_duplicates: Whether to skip duplicate chunks (default: True)
            duplicate_check_mode: How to check duplicates - "document_id", "docket_number", or "both"
            overwrite_collection: Whether to delete and recreate collection (default: False)
            
        Returns:
            str: Name of the created/updated Qdrant collection
        """
        logger.info(f"🔍 Processing {len(chunks)} semantic chunks for vector search")
        logger.info(f"Using model: {self.model_name}")
        
        # Filter duplicates if requested
        original_chunk_count = len(chunks)
        duplicates_found = 0
        
        if skip_duplicates and not overwrite_collection:
            chunks, duplicates_found, _ = self.filter_duplicate_chunks(chunks, duplicate_check_mode)
            if len(chunks) == 0:
                logger.info("✅ All chunks already exist in collection - nothing to process!")
                return self.collection_name
        
        # Check free tier limits if using cloud (after filtering duplicates)
        is_cloud = "cloud.qdrant.io" in self.qdrant_url or os.getenv("QDRANT_API_KEY")
        if is_cloud:
            logger.info("☁️ Cloud Qdrant detected - checking free tier limits")
        
        # Log initial memory usage
        initial_memory = get_memory_usage()
        logger.info(f"💾 Initial memory usage: {initial_memory['ram_mb']:.1f}MB RAM ({initial_memory['ram_percent']:.1f}%)")
        
        # Handle collection creation/deletion
        collection_existed = self.client.collection_exists(collection_name=self.collection_name)
        
        if overwrite_collection and collection_existed:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"🗑️ Deleted existing collection '{self.collection_name}' (overwrite mode)")
            collection_existed = False

        # Create collection if it doesn't exist
        if not collection_existed:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, 
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"📁 Created collection '{self.collection_name}' with vector size {self.vector_size}")
        else:
            logger.info(f"📁 Using existing collection '{self.collection_name}'")

        # Process chunks and create embeddings
        points = []
        processed_count = 0
        error_count = 0
        
        for chunk in chunks:
            chunk_text = chunk.get("text")
            if not chunk_text or len(chunk_text.strip()) < 10:
                error_count += 1
                continue
            
            try:
                # Create enhanced text representation
                enhanced_text = self._create_enhanced_text(chunk)
                
                # Create dense vector embedding with BGE optimization
                vector_embedding = self.embedder.encode(enhanced_text).tolist()
                
                # Prepare enhanced payload with legal metadata
                enhanced_payload = {
                    # Core identifiers
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "docket_number": chunk.get("docket_number"),
                    "case_name": chunk.get("case_name"),
                    "court_id": chunk.get("court_id"),
                    
                    # Chunk metadata
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk_text,
                    "token_count": chunk.get("token_count", 0),
                    "sentence_count": chunk.get("sentence_count", 0),
                    
                    # Legal analysis
                    "semantic_topic": chunk.get("semantic_topic", "general"),
                    "legal_importance_score": chunk.get("legal_importance_score", 0.0),
                    "keyword_density": chunk.get("keyword_density", 0.0),
                    "citation_count": chunk.get("citation_count", 0),
                    "citations_in_chunk": chunk.get("citations_in_chunk", []),
                    "chunk_confidence": chunk.get("chunk_confidence", 0.0),
                    
                    # Author and type info
                    "author": chunk.get("author", ""),
                    "opinion_type": chunk.get("opinion_type", ""),
                    "date_filed": chunk.get("date_filed"),
                    
                    # Processing metadata
                    "embedding_model": self.model_name,
                    "enhanced_text": enhanced_text,
                    "processed_at": datetime.now().isoformat()
                }
                
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),  # Always use a valid UUID string
                        vector=vector_embedding,
                        payload=enhanced_payload
                    )
                )
                processed_count += 1
                
                # Process in batches to avoid memory issues
                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection_name, 
                        points=points, 
                        wait=True
                    )
                    logger.info(f"  Uploaded batch of {len(points)} chunks...")
                    points = []
                    
                    # Force garbage collection and memory monitoring at checkpoint intervals
                    if processed_count % checkpoint_interval == 0:
                        gc.collect()
                        
                        # Log memory usage and progress at checkpoints
                        current_memory = get_memory_usage()
                        logger.info(f"💾 Checkpoint {processed_count}: Memory usage: {current_memory['ram_mb']:.1f}MB RAM ({current_memory['ram_percent']:.1f}%)")
                    
                if processed_count % checkpoint_interval == 0:
                    logger.info(f"  Processed {processed_count} chunks so far...")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                error_count += 1
                continue

        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name, 
                points=points, 
                wait=True
            )
            logger.info(f"  Uploaded final batch of {len(points)} chunks")
            
        # Final cleanup
        gc.collect()

        logger.info(f"✅ Vector processing completed!")
        logger.info(f"📊 Processing summary:")
        logger.info(f"   Original chunks: {original_chunk_count}")
        if duplicates_found > 0:
            logger.info(f"   Duplicates skipped: {duplicates_found}")
        logger.info(f"   New chunks processed: {processed_count}")
        logger.info(f"   Failed to process: {error_count}")
        logger.info(f"🏪 Updated Qdrant collection: {self.collection_name}")
        
        # Log cloud usage information
        if is_cloud:
            current_size_mb = self.get_collection_size_mb()
            logger.info(f"☁️ Cloud storage used: {current_size_mb:.1f}MB / 1024MB (free tier)")
            remaining_mb = 1024.0 - current_size_mb
            logger.info(f"💾 Remaining free tier storage: {remaining_mb:.1f}MB")
        
        return self.collection_name
    
    
    def process_documents_from_file(self, chunks_file: str, 
                                   batch_size: int = 50, 
                                   checkpoint_interval: int = 100,
                                   skip_duplicates: bool = True,
                                   duplicate_check_mode: str = "document_id",
                                   overwrite_collection: bool = False) -> str:
        """
        Process chunks from JSON file with duplicate detection.
        
        Args:
            chunks_file: Path to JSON file with semantic chunks
            batch_size: Number of chunks to process in each batch (default: 50)
            checkpoint_interval: Interval for progress logging and cleanup (default: 100)
            skip_duplicates: Whether to skip duplicate chunks (default: True)
            duplicate_check_mode: How to check duplicates - "document_id", "docket_number", or "both"
            overwrite_collection: Whether to delete and recreate collection (default: False)
            
        Returns:
            Collection name
        """
        logger.info(f"📖 Loading chunks from {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"📊 Loaded {len(chunks)} chunks from file")
        return self.process_chunks(
            chunks, 
            batch_size=batch_size, 
            checkpoint_interval=checkpoint_interval,
            skip_duplicates=skip_duplicates,
            duplicate_check_mode=duplicate_check_mode,
            overwrite_collection=overwrite_collection
        )
    
    
    def semantic_search(self, query: str, collection_name: str = None, 
                       limit: int = 10, score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Perform semantic vector search using dense embeddings only.
        
        Args:
            query: Search query text
            collection_name: Name of the collection (uses self.collection_name if not provided)
            limit: Number of results to return
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of search results with scores and metadata
        """
        collection = collection_name or self.collection_name
        
        if not self.client.collection_exists(collection):
            raise ValueError(f"Collection '{collection}' does not exist.")
        
        logger.info(f"🔍 Performing semantic search on '{collection}'")
        logger.info(f"📊 Query: {query}")
        
        # Create enhanced query with BGE prefix if applicable
        enhanced_query = self.query_prefix + query if self.query_prefix else query
        
        # Create dense vector for the query
        query_vector = self.embedder.encode(enhanced_query).tolist()
        
        # Perform vector search
        try:
            search_result = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for point in search_result:
                result = {
                    'id': point.id,
                    'score': point.score,
                    'payload': point.payload,
                    'search_type': 'semantic'
                }
                results.append(result)
            
            logger.info(f"✅ Semantic search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            raise


# This main function can be run in isolation to understand how the processor works
def main():
    """Command line interface for chunk vector processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create vector embeddings for legal document chunks')
    parser.add_argument('chunks_file', help='Input JSON file with semantic chunks')
    parser.add_argument('--model', default='BAAI/bge-small-en-v1.5', help='Embedding model to use')
    parser.add_argument('--collection', default='caselaw-chunks-vector', help='Qdrant collection name')
    parser.add_argument('--qdrant_url', default='http://localhost:6333', help='Qdrant server URL')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedVectorProcessor(
        model_name=args.model,
        collection_name=args.collection,
        qdrant_url=args.qdrant_url
    )
    
    # Process chunks
    collection_name = processor.process_documents_from_file(args.chunks_file)
    
    print(f"\n🎉 Vector processing complete!")
    print(f"Collection: {collection_name}")
    print(f"Ready for hybrid search queries!")


if __name__ == "__main__":
    main()