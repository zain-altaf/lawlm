"""
Vector Processor for dense vector embeddings.

This processor handles the creation of dense vector embeddings using
sentence transformers. These embeddings are shared between pure vector
search and hybrid search methods.
"""

import uuid
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()


class VectorProcessor:
    """Processes documents for vector search using dense embeddings."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 collection_name: str = "caselaw-vector-search",
                 qdrant_url: str = None):
        """
        Initialize the vector processor.
        
        Args:
            model_name: Name of the sentence transformer model
            collection_name: Name of the Qdrant collection
            qdrant_url: URL of the Qdrant server
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.vector_size = 384  # Size for all-MiniLM-L6-v2
        
        # Load the embedding model
        print(f"Loading embedding model: {model_name}")
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
    
    def process(self, documents: List[Dict[str, Any]]) -> str:
        """
        Process documents for vector search.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            str: Name of the created Qdrant collection
        """
        print(f"\n--- Processing for Vector Search ({self.model_name}) ---")
        
        # Delete existing collection if it exists
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted existing collection '{self.collection_name}'")

        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size, 
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection '{self.collection_name}'")

        # Process documents and create embeddings
        points = []
        processed_count = 0
        
        for doc in documents:
            opinion_text = doc.get("opinion_text")
            if not opinion_text:
                continue
            
            # Create dense vector embedding
            try:
                vector_embedding = self.embedder.encode(opinion_text).tolist()
                
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector_embedding,
                        payload=doc
                    )
                )
                processed_count += 1
                
                # Process in batches to avoid memory issues
                if len(points) >= 100:
                    self.client.upsert(
                        collection_name=self.collection_name, 
                        points=points, 
                        wait=True
                    )
                    points = []
                    
            except Exception as e:
                print(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                continue

        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name, 
                points=points, 
                wait=True
            )

        print(f"Processed {processed_count} documents for vector search")
        print(f"Created Qdrant collection: {self.collection_name}")
        
        return self.collection_name
    
    def get_search_config(self) -> Dict[str, Any]:
        """
        Get configuration for vector search.
        
        Returns:
            dict: Configuration for vector search
        """
        return {
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "vector_size": self.vector_size,
            "qdrant_url": self.qdrant_url
        }
    
    def get_embedder(self) -> SentenceTransformer:
        """
        Get the loaded embedding model for reuse.
        
        Returns:
            SentenceTransformer: The loaded embedding model
        """
        return self.embedder