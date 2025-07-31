"""
Processing module for legal case search system.

This module contains specialized processors for different search methods:
- KeywordProcessor: Handles TF-IDF keyword search processing
- VectorProcessor: Handles dense vector embeddings for semantic search
- HybridProcessor: Handles combined dense + sparse vector processing

All processors share common utilities and configurations.
"""

from .keyword_processor import KeywordProcessor
from .vector_processor import VectorProcessor
from .hybrid_processor import HybridProcessor

__all__ = ['KeywordProcessor', 'VectorProcessor', 'HybridProcessor']