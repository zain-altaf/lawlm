"""
Processing module for legal document pipeline.

This module contains processors for the legal document processing pipeline:
- smart_chunking: Semantic chunking with Legal BERT
- vector_processor: Dense vector embeddings for semantic search

The pipeline focuses on ingestion, chunking, and vector processing.
"""

# Main processing components are imported directly by their module names
# e.g., from processing.smart_chunking import SemanticChunker
# e.g., from processing.vector_processor import EnhancedVectorProcessor

__all__ = ['smart_chunking', 'vector_processor']