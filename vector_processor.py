"""
Enhanced Vector Processor for semantic chunks with improved embeddings.

This processor handles the creation of dense vector embeddings using
BAAI/bge-small-en-v1.5 for better legal domain performance. Optimized
for semantically chunked legal documents.
"""

from http import client
import uuid
import os
import json
import logging
import gc
import subprocess
import time
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
            
            if api_key and "cloud.qdrant.io" in self.qdrant_url:
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
    

    def get_existing_docket_ids(self) -> set:
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
                    docket = point.payload.get('docket_id')
                    if docket:
                        existing_dockets.add(docket)
                
                offset = next_offset
                if next_offset is None:
                    break
            
            return existing_dockets
            
        except Exception as e:
            logger.warning(f"Could not get existing docket numbers: {e}")
            return set()
    
    
    def extract_legal_info(self, text: str) -> Dict[str, Any]:
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


    def clean_text(self, content: str) -> str:
        """Strips HTML/XML tags and normalizes whitespace."""
        if not content:
            return ''
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)


    def enhanced_text_processing(self, text: str) -> Dict[str, Any]:
        """Enhanced processing that extracts citations and legal entities."""
        if not text:
            return {
                'cleaned_text': '',
                'citations': [],
                'legal_entities': {'judges': [], 'parties': [], 'courts': [], 'statutes': []},
                'text_stats': {'length': 0, 'word_count': 0}
            }
        
        cleaned = self.clean_text(text)
        legal_info = self.extract_legal_info(cleaned)
        
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


    def process_and_upload_chunks(self, chunks: List[Dict[str, Any]], 
                                collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process and upload a batch of chunks for a single docket to Qdrant.
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
                    vectors_config={
                        'bge-small': models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        'bm25': models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        ),
                    },
                )
            
            # Create enhanced texts and embeddings for chunks
            enhanced_texts = []
            for chunk in chunks:
                processed = self.enhanced_text_processing(chunk['text'])
                enhanced_text = processed['cleaned_text']
                chunk['extracted_citations'] = processed['citations']
                chunk['extracted_entities'] = processed['legal_entities']
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
                    'id': chunk.get('id'),
                    'chunk_id': chunk.get('chunk_id'),
                    'docket_id': chunk.get('docket_id', ''),
                    'cluster_id': chunk.get('cluster_id', ''),
                    'opinion_id': chunk.get('opinion_id', ''),
                    'docket_number': chunk.get('docket_number', ''),
                    'case_name': chunk.get('case_name', ''),
                    'court_id': chunk.get('court_id', ''),
                    'chunk_index': chunk.get('chunk_index'),
                    'text': chunk.get('text', ''),
                    'author': chunk.get('author', ''),
                    'opinion_type': chunk.get('opinion_type', ''),
                    'date_filed': chunk.get('date_filed', ''),
                    'citations': chunk.get('citations', []),
                    'legal_entities': chunk.get('legal_entities', {}),
                    'citation_count': len(chunk.get('citations', [])),
                    'precedential_status': chunk.get('precedential_status', ''),
                    'sha1': chunk.get('sha1', ''),
                    'download_url': chunk.get('download_url', ''),
                    'source_field': chunk.get('source_field', ''),
                    'judges': chunk.get('judges', ''),
                    'date_created': chunk.get('date_created', ''),
                    'date_modified': chunk.get('date_modified', '')
                }

                
                points.append(models.PointStruct(
                    id=point_id,
                    vector={
                        "bge-small": embedding.tolist(),
                        "bm25": models.Document(
                            text=enhanced_texts[i],
                            model="Qdrant/bm25"
                        ),
                    },
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
    
    def hybrid_search(self, query: str, collection_name: str = None, 
                     limit: int = 10, score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF) combining dense and sparse vectors.
        
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
        
        logger.info(f"🔍 Performing hybrid search on '{collection}'")
        logger.info(f"📊 Query: {query}")
        
        # Create enhanced query with BGE prefix if applicable
        enhanced_query = self.query_prefix + query if self.query_prefix else query
        
        # Create dense vector for the query
        query_vector = self.embedder.encode(enhanced_query).tolist()
        
        try:
            # Perform hybrid search with RRF using prefetch
            search_result = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    # Dense vector search (semantic)
                    models.Prefetch(
                        query=query_vector,
                        using="bge-small",
                        limit=(5 * limit),  # Fetch more results for better fusion
                    ),
                    # Sparse vector search (keyword/BM25)
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model="Qdrant/bm25",
                        ),
                        using="bm25",
                        limit=(5 * limit),  # Fetch more results for better fusion
                    ),
                ],
                # Use RRF to combine the results
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for point in search_result.points:
                result = {
                    'id': point.id,
                    'score': point.score,
                    'payload': point.payload,
                    'search_type': 'hybrid_rrf'
                }
                results.append(result)
            
            logger.info(f"✅ Hybrid search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise
    
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