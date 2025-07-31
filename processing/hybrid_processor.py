"""
Hybrid Processor for combined dense + sparse vector search.

This processor handles the creation of hybrid search indices that combine:
- Dense vectors: Shared embeddings from VectorProcessor
- Sparse vectors: BM25-based sparse representations

This enables hybrid search with Reciprocal Rank Fusion (RRF).
"""

import uuid
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()


class HybridProcessor:
    """Processes documents for hybrid search using dense + sparse vectors."""
    
    def __init__(self, 
                 embedder: SentenceTransformer = None,
                 collection_name: str = "caselaw-hybrid-search",
                 sparse_model: str = "Qdrant/bm25",
                 chunk_size: int = 500,
                 qdrant_url: str = None):
        """
        Initialize the hybrid processor.
        
        Args:
            embedder: Pre-loaded SentenceTransformer model (shared from VectorProcessor)
            collection_name: Name of the Qdrant collection
            sparse_model: Name of the sparse model (BM25)
            chunk_size: Maximum words per chunk
            qdrant_url: URL of the Qdrant server
        """
        self.collection_name = collection_name
        self.sparse_model = sparse_model
        self.chunk_size = chunk_size
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.vector_size = 384  # Size for all-MiniLM-L6-v2
        
        # Use shared embedder or load new one
        if embedder is not None:
            self.embedder = embedder
            print("Using shared embedding model for hybrid search")
        else:
            model_name = 'all-MiniLM-L6-v2'
            print(f"Loading embedding model for hybrid search: {model_name}")
            self.embedder = SentenceTransformer(model_name)
            print("Embedding model loaded successfully")
        
        # Initialize Qdrant client
        self.client = self._get_qdrant_client()
    
    def _get_qdrant_client(self) -> QdrantClient:
        """Initialize and return Qdrant client."""
        try:
            client = QdrantClient(self.qdrant_url)
            client.get_collections()
            print(f"Connected to Qdrant at {self.qdrant_url}")
            return client
        except Exception as e:
            print(f"Error connecting to Qdrant at {self.qdrant_url}")
            print(f"Details: {e}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of maximum word count."""
        words = text.split()
        return [' '.join(words[i:i+self.chunk_size]) 
                for i in range(0, len(words), self.chunk_size)]
    
    def process(self, documents: List[Dict[str, Any]]) -> str:
        """
        Process documents for hybrid search.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            str: Name of the created Qdrant collection
        """
        print(f"\n--- Processing for Hybrid Search (Dense + {self.sparse_model}) ---")
        
        # Delete existing collection if it exists
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted existing collection '{self.collection_name}'")

        # Create hybrid collection with dense + sparse vectors
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.vector_size, 
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        )
        print(f"Created hybrid collection '{self.collection_name}'")

        # Process documents with chunking
        points = []
        processed_chunks = 0
        
        for doc in documents:
            opinion_text = doc.get("opinion_text")
            if not opinion_text:
                continue
            
            # Split document into chunks for better hybrid search
            chunks = self._chunk_text(opinion_text)
            
            for chunk in chunks:
                try:
                    # Create chunk payload with original document metadata
                    chunk_payload = {**doc, "opinion_text": chunk}
                    
                    # Create dense vector embedding (shared with vector search)
                    dense_vector = self.embedder.encode(chunk).tolist()

                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector={
                                "dense": dense_vector,
                                "sparse": models.Document(text=chunk, model=self.sparse_model)
                            },
                            payload=chunk_payload
                        )
                    )
                    processed_chunks += 1
                    
                    # Process in batches to avoid memory issues
                    if len(points) >= 50:  # Smaller batches for hybrid due to complexity
                        self.client.upsert(
                            collection_name=self.collection_name, 
                            points=points, 
                            wait=True
                        )
                        points = []
                        
                except Exception as e:
                    print(f"Error processing chunk from document {doc.get('id', 'unknown')}: {e}")
                    continue

        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name, 
                points=points, 
                wait=True
            )

        print(f"Processed {processed_chunks} document chunks for hybrid search")
        print(f"Created Qdrant collection: {self.collection_name}")
        
        return self.collection_name
    
    def get_search_config(self) -> Dict[str, Any]:
        """
        Get configuration for hybrid search.
        
        Returns:
            dict: Configuration for hybrid search
        """
        return {
            "collection_name": self.collection_name,
            "sparse_model": self.sparse_model,
            "chunk_size": self.chunk_size,
            "vector_size": self.vector_size,
            "qdrant_url": self.qdrant_url
        }