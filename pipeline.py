#!/usr/bin/env python3
"""
Unified Legal Document Processing Pipeline

This module orchestrates the complete pipeline:
1. Data ingestion from CourtListener API
2. Chunking (simple, not semantic)
3. Vector processing with BGE embeddings
4. Hybrid search index creation

Provides a single entry point for the entire processing workflow.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging
import argparse
import json
import re
import uuid

# Import our processing modules
from fetch_and_process import process_docket, process_docket_in_batches
from config import PipelineConfig, load_config
from batch_utils import BatchProcessor, create_job_id
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Handle vector processor import gracefully
try:
    from processing.hybrid_indexer import EnhancedVectorProcessor
    VECTOR_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vector processing not available ({e}). Only chunking will work.")
    EnhancedVectorProcessor = None
    VECTOR_PROCESSOR_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    def run_full_pipeline(self, use_existing_raw: bool = False) -> Dict[str, Any]:
        logger.info(f"🚀 Starting full legal document processing pipeline")
        logger.info(f"🏛️ Court: {self.config.data_ingestion.court}, Dockets: {self.config.data_ingestion.num_dockets}")
        pipeline_start = datetime.now()
        results = {}

        try:
            raw_docs_path = self.working_dir / "raw_cases.json"
            if use_existing_raw and raw_docs_path.exists():
                logger.info(f"📄 Loading existing raw documents from: {raw_docs_path}")
                with open(raw_docs_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
            else:
                # Ingestion
                docs = process_docket(
                    court=self.config.data_ingestion.court,
                    num_dockets=self.config.data_ingestion.num_dockets
                )
                with open(raw_docs_path, "w", encoding="utf-8") as f:
                    json.dump(docs, f, ensure_ascii=False, indent=2)
                logger.info(f"📄 Raw documents saved to: {raw_docs_path}")

            # Chunking step - convert raw documents to chunks
            chunks_path = self.working_dir / "semantic_chunks.json"
            if not chunks_path.exists():
                # Create LangChain splitter optimized for legal documents with paragraph preservation
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.semantic_chunking.target_chunk_size * 4,  # Convert tokens to chars
                    chunk_overlap=self.config.semantic_chunking.overlap_size * 4,
                    length_function=len,
                    separators=[
                        "\n\n\n",  # Major section breaks
                        "\n\n",    # Paragraph breaks (prioritized to keep paragraphs together)
                        "\n",      # Line breaks
                        ". ",      # Sentence endings
                        "; ",      # Clause separators
                        ", ",      # Phrase separators
                        " ",       # Word boundaries
                        ""         # Character level (fallback)
                    ]
                )

                logger.info(f"🔪 Chunking {len(docs)} documents with LangChain RecursiveCharacterTextSplitter")
                logger.info(f"   Chunk size: {self.config.semantic_chunking.target_chunk_size * 4} chars")
                logger.info(f"   Overlap: {self.config.semantic_chunking.overlap_size * 4} chars")
                logger.info(f"   Prioritizing paragraph preservation")

                all_chunks = []
                for doc_idx, doc in enumerate(docs):
                    opinion_text = doc.get('opinion_text', '')
                    if not opinion_text or len(opinion_text.strip()) < 50:
                        continue

                    # Split the document text
                    text_chunks = text_splitter.split_text(opinion_text)

                    # Convert to our chunk format with metadata
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        if len(chunk_text.strip()) < self.config.semantic_chunking.min_chunk_size * 4:
                            continue

                        chunk = {
                            "chunk_id": (doc_idx * 10000) + chunk_idx,
                            "document_id": str(doc.get('id', f'doc_{doc_idx}')),
                            "docket_number": doc.get('docket_number', ''),
                            "case_name": doc.get('case_name', ''),
                            "court_id": doc.get('court_id', ''),
                            "chunk_index": chunk_idx,
                            "text": chunk_text.strip(),
                            "token_count": len(chunk_text.split()),
                            "sentence_count": len([s for s in chunk_text.split('.') if s.strip()]),
                            "semantic_topic": "general",
                            "legal_importance_score": 0.5,
                            "keyword_density": 0.0,
                            "citation_count": len(re.findall(r'\d+\s+[A-Z][a-z\.]*\s+\d+', chunk_text)),
                            "citations_in_chunk": [],
                            "chunk_confidence": 1.0,
                            "author": doc.get('author', ''),
                            "opinion_type": doc.get('opinion_type', ''),
                            "date_filed": doc.get('date_filed', '')
                        }
                        all_chunks.append(chunk)

                # Save chunks
                with open(chunks_path, 'w', encoding='utf-8') as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                logger.info(f"📁 Saved {len(all_chunks)} chunks to: {chunks_path}")
            else:
                logger.info(f"📁 Loading existing chunks from: {chunks_path}")
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    all_chunks = json.load(f)

            # Hybrid indexing (combines dense semantic vectors + sparse keyword vectors)
            if VECTOR_PROCESSOR_AVAILABLE and self.config.vector_processing.collection_name_vector:
                vector_processor = EnhancedVectorProcessor(
                    model_name=self.config.vector_processing.embedding_model,
                    collection_name=self.config.vector_processing.collection_name_vector,
                    qdrant_url=self.config.qdrant.url
                )
                hybrid_collection = self.config.vector_processing.collection_name_vector
                hybrid_results = vector_processor.create_hybrid_index(
                    all_chunks,
                    collection_name=hybrid_collection,
                    batch_size=self.config.vector_processing.batch_size,
                    checkpoint_interval=self.config.batch_processing.checkpoint_interval
                )
                results['hybrid'] = hybrid_results
                logger.info(f"🔍 Hybrid collection: {hybrid_collection}")
                logger.info(f"   Dense vectors: Semantic similarity search")
                logger.info(f"   Sparse vectors: Keyword/phrase matching")
                logger.info(f"   Search capability: Combined semantic + keyword via RRF")

            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
            summary_path = self.working_dir / "pipeline_summary.json"
            summary = {
                'pipeline_completed_at': datetime.now().isoformat(),
                'total_duration_seconds': pipeline_duration,
                'configuration': self.config.get_summary(),
                'results': results
            }
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"📋 Pipeline summary saved to: {summary_path}")

            return results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and available data."""
        status = {
            'working_dir': str(self.working_dir),
            'files': [str(f) for f in self.working_dir.glob("*")],
            'configuration': self.config.dict()
        }
        return status

def main():
    parser = argparse.ArgumentParser(
        description="Unified Legal Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', help='Configuration file path (JSON)')
    parser.add_argument('--status', action='store_true', help='Show pipeline status and exit')
    parser.add_argument('--use-existing-raw', action='store_true', help='Use existing raw_cases.json and skip ingestion')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config) if args.config else load_config()

    pipeline = LegalDocumentPipeline(config=config)

    if args.status:
        status = pipeline.get_pipeline_status()
        print(json.dumps(status, indent=2))
        return

    pipeline.run_full_pipeline(use_existing_raw=args.use_existing_raw)

if __name__ == "__main__":
    main()