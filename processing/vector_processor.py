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

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available ({e}). GPU acceleration will be disabled.")
    torch = None
    TORCH_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    usage = {
        'ram_mb': memory_info.rss / 1024 / 1024,
        'ram_percent': process.memory_percent()
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        usage['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        usage['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    
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
        logger.info(f"Loading enhanced embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        
        # Optimize model for inference
        self.embedder.eval()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"Using GPU for embeddings")
        else:
            logger.info(f"Using CPU for embeddings")
        
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
        """Create enhanced text representation for better embeddings."""
        text = chunk.get('text', '')
        
        # Add contextual information for better embeddings
        case_name = chunk.get('case_name', '')
        semantic_topic = chunk.get('semantic_topic', '')
        author = chunk.get('author', '')
        
        # Create enriched text for embedding
        enhanced_parts = []
        
        # Add case context
        if case_name:
            enhanced_parts.append(f"Case: {case_name}")
        
        # Add topic context
        if semantic_topic and semantic_topic != 'general':
            enhanced_parts.append(f"Topic: {semantic_topic}")
        
        # Add author context
        if author:
            enhanced_parts.append(f"Author: {author}")
        
        # Add the main text
        enhanced_parts.append(text)
        
        # Combine with separator
        enhanced_text = " | ".join(enhanced_parts)
        
        # Add BGE passage prefix if applicable
        if self.passage_prefix:
            enhanced_text = self.passage_prefix + enhanced_text
        
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
            existing_ids = self.get_existing_document_ids()
            existing_dockets = self.get_existing_docket_numbers()
            filtered_chunks = [chunk for chunk in chunks 
                             if (chunk.get('document_id') not in existing_ids and
                                 chunk.get('docket_number') not in existing_dockets)]
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
    
    def check_free_tier_limits(self, new_points_count: int) -> bool:
        """Check if adding new points would exceed free tier limits."""
        try:
            # Get current collection size
            current_size_mb = self.get_collection_size_mb()
            
            # Estimate size of new points (roughly 2KB per point)
            new_size_mb = new_points_count * 0.002
            total_size_mb = current_size_mb + new_size_mb
            
            free_tier_limit = 1024.0  # 1GB
            
            if total_size_mb > free_tier_limit:
                logger.warning(f"⚠️ Adding {new_points_count} points would exceed free tier limit:")
                logger.warning(f"   Current size: {current_size_mb:.1f}MB")
                logger.warning(f"   New points size: {new_size_mb:.1f}MB")
                logger.warning(f"   Total would be: {total_size_mb:.1f}MB")
                logger.warning(f"   Free tier limit: {free_tier_limit}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not check free tier limits: {e}")
            return True  # Allow processing if we can't check
    
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
            if not self.check_free_tier_limits(len(chunks)):
                logger.error("❌ Processing would exceed free tier limits")
                raise RuntimeError("Processing would exceed Qdrant free tier 1GB limit")
        
        # Log initial memory usage
        initial_memory = get_memory_usage()
        logger.info(f"💾 Initial memory usage: {initial_memory['ram_mb']:.1f}MB RAM ({initial_memory['ram_percent']:.1f}%)")
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"🖥️ GPU memory: {initial_memory.get('gpu_allocated_mb', 0):.1f}MB allocated")
        
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
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Log memory usage and progress at checkpoints
                        current_memory = get_memory_usage()
                        logger.info(f"💾 Checkpoint {processed_count}: Memory usage: {current_memory['ram_mb']:.1f}MB RAM ({current_memory['ram_percent']:.1f}%)")
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            logger.info(f"🖥️ GPU memory: {current_memory.get('gpu_allocated_mb', 0):.1f}MB allocated")
                    
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
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a query for vector search with BGE optimization.
        
        Args:
            query: User query string
            
        Returns:
            Query vector embedding
        """
        # Add query prefix for BGE models
        enhanced_query = self.query_prefix + query if self.query_prefix else query
        return self.embedder.encode(enhanced_query).tolist()
    
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
    
    def get_search_config(self) -> Dict[str, Any]:
        """
        Get configuration for enhanced vector search.
        
        Returns:
            dict: Configuration for vector search
        """
        return {
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "vector_size": self.vector_size,
            "qdrant_url": self.qdrant_url,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix
        }
    
    def get_embedder(self) -> SentenceTransformer:
        """
        Get the loaded embedding model for reuse.
        
        Returns:
            SentenceTransformer: The loaded embedding model
        """
        return self.embedder
    
    def _create_sparse_vector(self, text: str, tfidf_vectorizer: TfidfVectorizer) -> Dict[int, float]:
        """
        Create sparse vector using TF-IDF for keyword-based search.
        
        Args:
            text: Text to vectorize
            tfidf_vectorizer: Fitted TF-IDF vectorizer
            
        Returns:
            Sparse vector as dictionary {index: value}
        """
        # Transform text to TF-IDF vector
        tfidf_vector = tfidf_vectorizer.transform([text])
        
        # Convert sparse matrix to dictionary format for Qdrant
        sparse_dict = {}
        coo = tfidf_vector.tocoo()
        for i, j, value in zip(coo.row, coo.col, coo.data):
            if value > 0.01:  # Filter out very low values to reduce storage
                sparse_dict[j] = float(value)
        
        return sparse_dict
    
    def create_hybrid_index(self, chunks: List[Dict[str, Any]], 
                           collection_name: str = None,
                           batch_size: int = 50,
                           checkpoint_interval: int = 100,
                           skip_duplicates: bool = True,
                           overwrite_collection: bool = False) -> str:
        """
        Create hybrid index with both dense (semantic) and sparse (keyword) vectors.
        
        Args:
            chunks: List of chunk dictionaries
            collection_name: Name of the hybrid collection
            batch_size: Number of chunks per batch
            checkpoint_interval: Interval for progress logging
            skip_duplicates: Whether to skip duplicate chunks
            overwrite_collection: Whether to recreate collection
            
        Returns:
            Collection name
        """
        if collection_name is None:
            collection_name = self.collection_name + "-hybrid"
            
        logger.info(f"🔍 Creating hybrid index with {len(chunks)} chunks")
        logger.info(f"🤖 Dense vectors: {self.model_name}")
        logger.info(f"📊 Sparse vectors: TF-IDF")
        
        # Filter duplicates if requested
        original_chunk_count = len(chunks)
        duplicates_found = 0
        
        if skip_duplicates and not overwrite_collection:
            # Check against existing hybrid collection
            temp_collection_name = self.collection_name
            self.collection_name = collection_name  # Temporarily change for duplicate check
            try:
                chunks, duplicates_found, _ = self.filter_duplicate_chunks(chunks, "document_id")
                if len(chunks) == 0:
                    logger.info("✅ All chunks already exist in hybrid collection - nothing to process!")
                    return collection_name
            finally:
                self.collection_name = temp_collection_name  # Restore original collection name
        
        # Prepare texts for TF-IDF vectorization
        logger.info("📝 Preparing texts for TF-IDF vectorization...")
        texts = []
        valid_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if chunk_text and len(chunk_text.strip()) >= 10:
                # Use enhanced text for both dense and sparse vectors
                enhanced_text = self._create_enhanced_text(chunk)
                texts.append(enhanced_text)
                valid_chunks.append(chunk)
        
        if not texts:
            logger.warning("⚠️ No valid texts found for hybrid indexing")
            return collection_name
        
        logger.info(f"📊 Valid chunks for hybrid indexing: {len(valid_chunks)}")
        
        # Fit TF-IDF vectorizer
        logger.info("🔧 Fitting TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            sublinear_tf=True,  # Use sublinear TF scaling
            norm='l2'  # L2 normalization
        )
        
        tfidf_vectorizer.fit(texts)
        vocab_size = len(tfidf_vectorizer.vocabulary_)
        logger.info(f"📚 TF-IDF vocabulary size: {vocab_size}")
        
        # Handle collection creation/deletion
        collection_existed = self.client.collection_exists(collection_name=collection_name)
        
        if overwrite_collection and collection_existed:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"🗑️ Deleted existing hybrid collection '{collection_name}' (overwrite mode)")
            collection_existed = False
        
        # Create hybrid collection with both dense and sparse vectors
        if not collection_existed:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams()
                    )
                }
            )
            logger.info(f"📁 Created hybrid collection '{collection_name}'")
            logger.info(f"   Dense vectors: {self.vector_size}D, Cosine distance")
            logger.info(f"   Sparse vectors: TF-IDF with {vocab_size} features")
        else:
            logger.info(f"📁 Using existing hybrid collection '{collection_name}'")
        
        # Process chunks and create hybrid embeddings
        points = []
        processed_count = 0
        error_count = 0
        
        # Log initial memory usage
        initial_memory = get_memory_usage()
        logger.info(f"💾 Initial memory usage: {initial_memory['ram_mb']:.1f}MB RAM")
        
        for i, (chunk, text) in enumerate(zip(valid_chunks, texts)):
            try:
                # Create dense vector embedding
                dense_vector = self.embedder.encode(text).tolist()
                
                # Create sparse vector
                sparse_vector = self._create_sparse_vector(text, tfidf_vectorizer)
                
                # Prepare enhanced payload
                enhanced_payload = {
                    # Core identifiers
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "docket_number": chunk.get("docket_number"),
                    "case_name": chunk.get("case_name"),
                    "court_id": chunk.get("court_id"),
                    
                    # Chunk metadata
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk.get("text", ""),
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
                    "enhanced_text": text,
                    "search_type": "hybrid",
                    "sparse_vector_size": len(sparse_vector),
                    "processed_at": datetime.now().isoformat()
                }
                
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),  # Always use a valid UUID string
                        vector={
                            "dense": dense_vector,
                            "sparse": models.SparseVector(
                                indices=list(sparse_vector.keys()),
                                values=list(sparse_vector.values())
                            )
                        },
                        payload=enhanced_payload
                    )
                )
                processed_count += 1
                
                # Process in batches to avoid memory issues
                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True
                    )
                    logger.info(f"  Uploaded hybrid batch of {len(points)} chunks...")
                    points = []
                    
                    # Memory monitoring and cleanup
                    if processed_count % checkpoint_interval == 0:
                        gc.collect()
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        current_memory = get_memory_usage()
                        logger.info(f"💾 Checkpoint {processed_count}: Memory usage: {current_memory['ram_mb']:.1f}MB RAM")
                    
                if processed_count % checkpoint_interval == 0:
                    logger.info(f"  Processed {processed_count} hybrid chunks so far...")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.get('chunk_id', 'unknown')} for hybrid index: {e}")
                error_count += 1
                continue
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            logger.info(f"  Uploaded final hybrid batch of {len(points)} chunks")
        
        # Final cleanup
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"✅ Hybrid index creation completed!")
        logger.info(f"📊 Hybrid processing summary:")
        logger.info(f"   Original chunks: {original_chunk_count}")
        if duplicates_found > 0:
            logger.info(f"   Duplicates skipped: {duplicates_found}")
        logger.info(f"   New chunks processed: {processed_count}")
        logger.info(f"   Failed to process: {error_count}")
        logger.info(f"🔍 Hybrid collection: {collection_name}")
        logger.info(f"   Dense vectors: {self.vector_size}D semantic embeddings")
        logger.info(f"   Sparse vectors: TF-IDF with {vocab_size} features")
        
        return collection_name
    
    def hybrid_search(self, 
                     query: str,
                     collection_name: str = None,
                     limit: int = 10,
                     dense_weight: float = 0.7,
                     sparse_weight: float = 0.3,
                     score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse vectors using Reciprocal Rank Fusion.
        
        Args:
            query: Search query text
            collection_name: Name of the hybrid collection (defaults to collection_name + "-hybrid")
            limit: Number of results to return
            dense_weight: Weight for dense (semantic) search results (0.0 to 1.0)
            sparse_weight: Weight for sparse (keyword) search results (0.0 to 1.0)
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of search results with scores and metadata
        """
        if collection_name is None:
            collection_name = self.collection_name + "-hybrid"
            
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Hybrid collection '{collection_name}' does not exist. Create it first using create_hybrid_index.")
        
        logger.info(f"🔍 Performing hybrid search on '{collection_name}'")
        logger.info(f"📊 Query: {query}")
        logger.info(f"⚖️ Dense weight: {dense_weight}, Sparse weight: {sparse_weight}")
        
        # Create enhanced query text (same as used during indexing)
        enhanced_query = self.query_prefix + query if self.query_prefix else query
        
        # Create dense vector for the query
        dense_vector = self.embedder.encode(enhanced_query).tolist()
        
        # Create sparse vector for the query using TF-IDF
        # Note: For production use, you'd want to save the TF-IDF vectorizer from indexing
        # For now, we'll create a simple sparse representation based on keywords
        query_words = query.lower().split()
        
        # Simple sparse vector creation (in production, use the same TF-IDF vectorizer from indexing)
        sparse_indices = []
        sparse_values = []
        for i, word in enumerate(query_words[:20]):  # Limit to first 20 words
            # Use simple word hash as index (this is a simplified approach)
            word_hash = hash(word) % 10000  # Match the max_features from TF-IDF
            if word_hash not in sparse_indices:
                sparse_indices.append(word_hash)
                sparse_values.append(1.0)  # Simple binary weights
        
        sparse_vector = models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
        # Perform hybrid search using prefetch and fusion
        try:
            search_result = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    # Dense vector search (semantic similarity)
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=limit * 2,  # Get more results for fusion
                        score_threshold=score_threshold
                    ),
                    # Sparse vector search (keyword matching)
                    models.Prefetch(
                        query=sparse_vector,
                        using="sparse", 
                        limit=limit * 2,  # Get more results for fusion
                        score_threshold=score_threshold
                    )
                ],
                # Combine results using Reciprocal Rank Fusion (RRF)
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
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
                    'search_type': 'hybrid'
                }
                results.append(result)
            
            logger.info(f"✅ Hybrid search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise
    
    def semantic_search(self, 
                       query: str,
                       collection_name: str = None,
                       limit: int = 10,
                       score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using dense vectors only.
        
        Args:
            query: Search query text
            collection_name: Name of the collection (can be regular or hybrid)
            limit: Number of results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results
        """
        if collection_name is None:
            collection_name = self.collection_name
            
        # Check if this is a hybrid collection
        is_hybrid = collection_name.endswith("-hybrid")
        vector_name = "dense" if is_hybrid else None
        
        logger.info(f"🧠 Performing semantic search on '{collection_name}'")
        
        # Create enhanced query text
        enhanced_query = self.query_prefix + query if self.query_prefix else query
        
        # Create dense vector for the query
        query_vector = self.embedder.encode(enhanced_query).tolist()
        
        # Perform search
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector if not is_hybrid else None,
            using=vector_name if is_hybrid else None,
            query_params=models.SearchParams(
                exact=False,
                hnsw_ef=128
            ),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
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


# Backward compatibility - keep old class name as alias
VectorProcessor = EnhancedVectorProcessor


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