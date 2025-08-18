#!/usr/bin/env python3
"""
Unified Legal Document Processing Pipeline

This module orchestrates the complete pipeline:
1. Data ingestion from CourtListener API
2. Smart semantic chunking with legal BERT
3. Vector processing with BGE embeddings
4. Hybrid search index creation

Provides a single entry point for the entire processing workflow.
"""

import os
import json
import logging
import argparse
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Import our processing modules
from data_ingestion import process_docket, enhanced_text_processing, process_docket_in_batches
from processing.smart_chunking import SemanticChunker
from config import PipelineConfig, load_config
from batch_utils import BatchProcessor, create_job_id

# Handle vector processor import gracefully
try:
    from processing.vector_processor import EnhancedVectorProcessor
    VECTOR_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vector processing not available ({e}). Only chunking will work.")
    EnhancedVectorProcessor = None
    VECTOR_PROCESSOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalDocumentPipeline:
    """
    Unified pipeline for processing legal documents from ingestion to search-ready vectors.
    """
    
    def __init__(self, 
                 config: Optional[PipelineConfig] = None,
                 working_dir: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 chunking_model: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 min_chunk_size: Optional[int] = None,
                 qdrant_url: Optional[str] = None):
        """
        Initialize the legal document processing pipeline.
        
        Args:
            config: Pipeline configuration object (overrides individual parameters)
            working_dir: Directory for storing intermediate files (legacy parameter)
            embedding_model: Model for vector embeddings (legacy parameter)
            chunking_model: Legal BERT model for semantic chunking (legacy parameter)
            chunk_size: Target chunk size in tokens (legacy parameter)
            min_chunk_size: Minimum chunk size (legacy parameter)
            qdrant_url: Qdrant server URL (legacy parameter)
        """
        # Use provided config or load default
        if config is None:
            config = load_config()
        self.config = config
        
        # Override config with legacy parameters if provided
        if working_dir is not None:
            self.config.processing.working_directory = working_dir
        if embedding_model is not None:
            self.config.vector_processing.embedding_model = embedding_model
        if chunking_model is not None:
            # Note: chunking_model is now ignored as we use RecursiveCharacterTextSplitter
            pass
        if chunk_size is not None:
            self.config.semantic_chunking.target_chunk_size = chunk_size
        if min_chunk_size is not None:
            self.config.semantic_chunking.min_chunk_size = min_chunk_size
        if qdrant_url is not None:
            self.config.qdrant.url = qdrant_url
        
        # Set up working directory
        self.working_dir = Path(self.config.processing.working_directory)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (lazily loaded)
        self._chunker = None
        self._vector_processor = None
        
        logger.info(f"🏛️ Legal Document Pipeline initialized")
        logger.info(f"📂 Working directory: {self.working_dir}")
        logger.info(f"🤖 Embedding model: {self.config.vector_processing.embedding_model}")
        logger.info(f"📝 Text splitter: RecursiveCharacterTextSplitter")
        logger.info(f"🔧 Using configuration with {len(self.config.validate())} validation issues")
    
    @property
    def chunker(self) -> SemanticChunker:  # Keep the type for backward compatibility
        """Lazy load the text splitter (backward compatible as SemanticChunker)."""
        if self._chunker is None:
            config = self.config.semantic_chunking
            logger.info(f"📝 Loading RecursiveCharacterTextSplitter")
            # Use RecursiveCharacterTextSplitter with character-based sizing
            self._chunker = SemanticChunker(  # This is now an alias for RecursiveCharacterTextSplitter
                chunk_size=config.target_chunk_size * 4,  # Convert from token estimate to characters (~4 chars per token)
                chunk_overlap=config.overlap_size * 4,     # Convert overlap to characters
                min_chunk_size=config.min_chunk_size * 4,  # Convert min size to characters
                quality_threshold=config.quality_threshold
            )
        return self._chunker
    
    @property
    def vector_processor(self) -> EnhancedVectorProcessor:
        """Lazy load the vector processor."""
        if not VECTOR_PROCESSOR_AVAILABLE:
            raise RuntimeError("Vector processing is not available. Check PyTorch installation.")
            
        if self._vector_processor is None:
            config = self.config.vector_processing
            logger.info(f"🔍 Loading vector processor: {config.embedding_model}")
            self._vector_processor = EnhancedVectorProcessor(
                model_name=config.embedding_model,
                collection_name=config.collection_name_vector,
                qdrant_url=self.config.qdrant.url
            )
        return self._vector_processor
    
    def get_hybrid_processor(self):
        """Get hybrid processor only when explicitly needed (not used by default)."""
        if not VECTOR_PROCESSOR_AVAILABLE:
            logger.error("Hybrid processor not available - vector processing is disabled")
            return None
            
        try:
            from processing.hybrid_processor import EnhancedHybridProcessor
            config = self.config.vector_processing
            logger.info(f"⚡ Loading hybrid processor (sharing embedder)")
            # Share the embedding model to save memory
            embedder = self.vector_processor.get_embedder()
            return EnhancedHybridProcessor(
                embedder=embedder,
                collection_name="caselaw-chunks-hybrid",
                sparse_model="Qdrant/bm25",
                qdrant_url=self.config.qdrant.url
            )
        except ImportError as e:
            logger.error(f"Hybrid processor not available: {e}")
            return None
    
    def ingest_documents(self, 
                        court: str = "scotus", 
                        num_dockets: int = 5,
                        batch_size: Optional[int] = None,
                        output_filename: Optional[str] = None,
                        job_id: Optional[str] = None,
                        resume: bool = False) -> str:
        """
        Step 1: Ingest legal documents from CourtListener API.
        
        Args:
            court: Court identifier
            num_dockets: Number of dockets to fetch
            batch_size: Optional batch size (if None, processes all at once)
            output_filename: Custom output filename
            job_id: Optional job identifier for batch processing
            resume: Whether to resume from previous incomplete run
            
        Returns:
            Path to saved documents
        """
        logger.info(f"📥 Starting document ingestion")
        logger.info(f"🏛️ Court: {court}")
        logger.info(f"📊 Number of dockets: {num_dockets}")
        
        start_time = time.time()
        
        # Determine if we should use batch processing
        use_batch_processing = (
            batch_size is not None and 
            batch_size < num_dockets
        )
        
        if use_batch_processing:
            logger.info(f"🔢 Using batch processing: {batch_size} dockets per batch")
            
            # Use batch processing
            batch_results = process_docket_in_batches(
                court=court,
                num_dockets=num_dockets,
                batch_size=batch_size,
                working_dir=str(self.working_dir),
                job_id=job_id,
                resume=resume
            )
            
            if not batch_results['success']:
                raise RuntimeError(f"Batch ingestion failed: {batch_results.get('error', 'Unknown error')}")
            
            # Use the final merged file
            output_path = batch_results['final_file']
            documents_count = batch_results['total_documents']
            
        else:
            logger.info(f"📥 Using single-batch processing")
            
            # Use existing single-batch ingestion function
            documents = process_docket(court=court, num_dockets=num_dockets)
            
            if not documents:
                raise RuntimeError("No documents were successfully ingested")
            
            # Save documents
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"raw_cases_{court}_{timestamp}.json"
            
            output_path = self.working_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            
            output_path = str(output_path)
            documents_count = len(documents)
        
        duration = time.time() - start_time
        logger.info(f"✅ Document ingestion completed in {duration:.1f}s")
        logger.info(f"📄 Documents ingested: {documents_count}")
        logger.info(f"💾 Saved to: {output_path}")
        
        return output_path
    
    def chunk_documents(self, input_file: str, output_filename: Optional[str] = None) -> str:
        """
        Step 2: Create semantic chunks from legal documents.
        
        Args:
            input_file: Path to ingested documents JSON
            output_filename: Custom output filename
            
        Returns:
            Path to saved chunks
        """
        logger.info(f"🔄 Starting semantic chunking")
        logger.info(f"📂 Input: {input_file}")
        
        start_time = time.time()
        
        # Generate output filename if not provided
        if output_filename is None:
            input_path = Path(input_file)
            output_filename = f"{input_path.stem}_chunks.json"
        
        output_path = self.working_dir / output_filename
        
        # Use the semantic chunker
        result_path = self.chunker.process_documents(input_file, str(output_path))
        
        duration = time.time() - start_time
        logger.info(f"✅ Semantic chunking completed in {duration:.1f}s")
        logger.info(f"💾 Chunks saved to: {result_path}")
        
        return result_path
    
    def create_vector_index(self, chunks_file: str, 
                           collection_name: Optional[str] = None,
                           skip_duplicates: bool = True,
                           duplicate_check_mode: str = "document_id",
                           overwrite_collection: bool = False) -> str:
        """
        Step 3: Create vector search index from chunks with duplicate detection.
        
        Args:
            chunks_file: Path to chunks JSON file
            collection_name: Custom collection name
            skip_duplicates: Whether to skip duplicate documents (default: True)
            duplicate_check_mode: How to check duplicates - "document_id", "docket_number", or "both"
            overwrite_collection: Whether to delete and recreate collection (default: False)
            
        Returns:
            Name of created collection
        """
        if not VECTOR_PROCESSOR_AVAILABLE:
            logger.warning("⚠️ Vector processing not available. Skipping vector index creation.")
            return None
            
        logger.info(f"🔍 Starting vector index creation")
        logger.info(f"📂 Input: {chunks_file}")
        
        # Log duplicate handling settings
        if overwrite_collection:
            logger.info(f"🗑️ Overwrite mode: will delete and recreate collection")
        elif skip_duplicates:
            logger.info(f"🔄 Duplicate detection enabled: checking by {duplicate_check_mode}")
        else:
            logger.info(f"⚠️ Duplicate detection disabled: may create duplicates")
        
        start_time = time.time()
        
        if collection_name:
            self.vector_processor.collection_name = collection_name
        
        collection = self.vector_processor.process_documents_from_file(
            chunks_file,
            skip_duplicates=skip_duplicates,
            duplicate_check_mode=duplicate_check_mode,
            overwrite_collection=overwrite_collection
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Vector index creation completed in {duration:.1f}s")
        logger.info(f"🏪 Collection: {collection}")
        
        return collection
    
    def create_hybrid_index(self, chunks_file: str, collection_name: Optional[str] = None) -> str:
        """
        Step 4: Create hybrid search index from chunks (optional, resource-intensive).
        
        Args:
            chunks_file: Path to chunks JSON file
            collection_name: Custom collection name
            
        Returns:
            Name of created collection
        """
        logger.info(f"⚡ Starting hybrid index creation")
        logger.info(f"📂 Input: {chunks_file}")
        
        start_time = time.time()
        
        hybrid_processor = self.get_hybrid_processor()
        if hybrid_processor is None:
            raise RuntimeError("Hybrid processor not available")
        
        if collection_name:
            hybrid_processor.collection_name = collection_name
        
        collection = hybrid_processor.process_documents_from_file(chunks_file)
        
        duration = time.time() - start_time
        logger.info(f"✅ Hybrid index creation completed in {duration:.1f}s")
        logger.info(f"🏪 Collection: {collection}")
        
        return collection
    
    def run_full_pipeline(self,
                         court: str = "scotus",
                         num_dockets: int = 5,
                         batch_size: Optional[int] = None,
                         create_vector: bool = True,
                         create_hybrid: bool = False,  # Disabled by default
                         vector_collection: Optional[str] = None,
                         hybrid_collection: Optional[str] = None,
                         job_id: Optional[str] = None,
                         resume: bool = False,
                         skip_duplicates: bool = True,
                         duplicate_check_mode: str = "document_id",
                         overwrite_collection: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline from ingestion to search-ready indices with persistent storage.
        
        Args:
            court: Court identifier
            num_dockets: Number of dockets to fetch
            batch_size: Optional batch size for processing
            create_vector: Whether to create vector search index
            create_hybrid: Whether to create hybrid search index (disabled by default for resource optimization)
            vector_collection: Custom vector collection name
            hybrid_collection: Custom hybrid collection name
            job_id: Optional job identifier for batch processing
            resume: Whether to resume from previous incomplete run
            skip_duplicates: Whether to skip duplicate documents in persistent storage (default: True)
            duplicate_check_mode: How to check duplicates - "document_id", "docket_number", or "both"
            overwrite_collection: Whether to delete and recreate collection (default: False)
            
        Returns:
            Dictionary with paths and collection names
        """
        logger.info(f"🚀 Starting full legal document processing pipeline")
        logger.info(f"🏛️ Court: {court}, Dockets: {num_dockets}")
        if batch_size:
            logger.info(f"🔢 Batch size: {batch_size}")
        logger.info(f"🔍 Vector index: {create_vector}")
        if create_hybrid:
            logger.info(f"⚡ Hybrid index: enabled (resource-intensive)")
        else:
            logger.info(f"⚡ Hybrid index: disabled (use --enable-hybrid to enable)")
        
        pipeline_start = time.time()
        results = {}
        
        try:
            # Step 1: Ingest documents
            logger.info(f"\n" + "="*60)
            logger.info(f"📥 STEP 1: Document Ingestion")
            logger.info(f"="*60)
            
            raw_docs_path = self.ingest_documents(
                court=court, 
                num_dockets=num_dockets,
                batch_size=batch_size,
                job_id=job_id,
                resume=resume
            )
            results['raw_documents'] = raw_docs_path
            
            # Step 2: Create chunks with recursive character text splitter
            logger.info(f"\n" + "="*60)
            logger.info(f"🔄 STEP 2: Recursive Character Text Splitting")
            logger.info(f"="*60)
            
            chunks_path = self.chunk_documents(raw_docs_path)
            results['chunks'] = chunks_path
            
            # Step 3: Create search indices
            collections = {}
            
            if create_vector:
                logger.info(f"\n" + "="*60)
                logger.info(f"🔍 STEP 3A: Vector Index Creation")
                logger.info(f"="*60)
                
                vector_coll = self.create_vector_index(
                    chunks_path, 
                    vector_collection,
                    skip_duplicates=skip_duplicates,
                    duplicate_check_mode=duplicate_check_mode,
                    overwrite_collection=overwrite_collection
                )
                if vector_coll:
                    collections['vector'] = vector_coll
                    results['vector_collection'] = vector_coll
                else:
                    logger.warning("⚠️ Vector index creation was skipped due to missing dependencies")
            
            if create_hybrid:
                logger.info(f"\n" + "="*60)
                logger.info(f"⚡ STEP 3B: Hybrid Index Creation")
                logger.info(f"="*60)
                
                hybrid_coll = self.create_hybrid_index(chunks_path, hybrid_collection)
                collections['hybrid'] = hybrid_coll
                results['hybrid_collection'] = hybrid_coll
            
            results['collections'] = collections
            
            # Pipeline summary
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"\n" + "="*60)
            logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"="*60)
            logger.info(f"⏱️ Total duration: {pipeline_duration:.1f}s")
            logger.info(f"📄 Raw documents: {raw_docs_path}")
            logger.info(f"🔢 Chunks: {chunks_path}")
            if create_vector:
                logger.info(f"🔍 Vector collection: {collections.get('vector')}")
            if create_hybrid:
                logger.info(f"⚡ Hybrid collection: {collections.get('hybrid')}")
            logger.info(f"🏪 Qdrant URL: {self.config.qdrant.url}")
            
            # Save pipeline summary
            summary = {
                'pipeline_completed_at': datetime.now().isoformat(),
                'total_duration_seconds': pipeline_duration,
                'configuration': {
                    'court': court,
                    'num_dockets': num_dockets,
                    'batch_size': batch_size,
                    'embedding_model': self.config.vector_processing.embedding_model,
                    'splitter_type': 'RecursiveCharacterTextSplitter',
                    'chunk_size_chars': self.config.semantic_chunking.target_chunk_size * 4,
                    'qdrant_url': self.config.qdrant.url
                },
                'results': results
            }
            
            summary_path = self.working_dir / "pipeline_summary.json"
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
            'files': [],
            'collections': [],
            'configuration': {
                'embedding_model': self.config.vector_processing.embedding_model,
                'splitter_type': 'RecursiveCharacterTextSplitter',
                'chunk_size_chars': self.config.semantic_chunking.target_chunk_size * 4,
                'qdrant_url': self.config.qdrant.url
            }
        }
        
        # Check for files in working directory
        for file_path in self.working_dir.glob("*.json"):
            file_info = {
                'name': file_path.name,
                'path': str(file_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            status['files'].append(file_info)
        
        # Check Qdrant collections (if available)
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(self.config.qdrant.url)
            collections = client.get_collections()
            for collection in collections.collections:
                status['collections'].append({
                    'name': collection.name,
                    'vectors_count': collection.vectors_count,
                    'points_count': collection.points_count
                })
        except Exception as e:
            status['qdrant_error'] = str(e)
        
        return status


def main():
    """Command line interface for the unified pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified Legal Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for Supreme Court cases (with duplicate detection)
  python pipeline_runner.py --court scotus --num-dockets 10

  # Run pipeline without duplicate detection (may create duplicates)
  python pipeline_runner.py --court scotus --num-dockets 5 --no-skip-duplicates

  # Overwrite existing collection (destroys previous data)
  python pipeline_runner.py --court scotus --num-dockets 5 --overwrite-collection

  # Check duplicates by docket number instead of document ID
  python pipeline_runner.py --court scotus --num-dockets 5 --duplicate-check-mode docket_number

  # Run only ingestion and chunking
  python pipeline_runner.py --court scotus --no-vector --no-hybrid

  # Custom configuration with persistent storage
  python pipeline_runner.py --court scotus --chunk-size 512 --embedding-model all-MiniLM-L6-v2

  # Check pipeline status
  python pipeline_runner.py --status
        """
    )
    
    # Main operation mode
    parser.add_argument('--status', action='store_true',
                       help='Show pipeline status and exit')
    parser.add_argument('--config', 
                       help='Configuration file path (JSON)')
    
    # Ingestion parameters
    parser.add_argument('--court', default='scotus',
                       help='Court identifier (default: scotus)')
    parser.add_argument('--num-dockets', type=int, default=5,
                       help='Number of dockets to fetch (default: 5)')
    parser.add_argument('--batch-size', type=int,
                       help='Process in batches of this size (enables batch processing)')
    parser.add_argument('--job-id',
                       help='Job identifier for batch processing (auto-generated if not provided)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous incomplete batch processing job')
    
    # Processing parameters
    parser.add_argument('--working-dir', default='data',
                       help='Working directory for files (default: data)')
    parser.add_argument('--embedding-model', default='BAAI/bge-small-en-v1.5',
                       help='Embedding model (default: BAAI/bge-small-en-v1.5)')
    parser.add_argument('--chunking-model', default='nlpaueb/legal-bert-base-uncased',
                       help='Legal BERT model for chunking')
    parser.add_argument('--chunk-size', type=int, default=384,
                       help='Target chunk size in tokens (default: 384)')
    parser.add_argument('--min-chunk-size', type=int, default=100,
                       help='Minimum chunk size (default: 100)')
    
    # Index creation options
    parser.add_argument('--no-vector', action='store_true',
                       help='Skip vector index creation')
    parser.add_argument('--enable-hybrid', action='store_true',
                       help='Enable hybrid index creation (resource intensive, disabled by default)')
    parser.add_argument('--vector-collection',
                       help='Custom vector collection name')
    parser.add_argument('--hybrid-collection',
                       help='Custom hybrid collection name')
    
    # Duplicate handling options
    parser.add_argument('--no-skip-duplicates', action='store_true',
                       help='Disable duplicate detection (may create duplicates in persistent storage)')
    parser.add_argument('--duplicate-check-mode', default='document_id',
                       choices=['document_id', 'docket_number', 'both'],
                       help='How to check for duplicates (default: document_id)')
    parser.add_argument('--overwrite-collection', action='store_true',
                       help='Delete and recreate collection (WARNING: destroys existing data)')
    
    # Infrastructure
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                       help='Qdrant server URL (default: http://localhost:6333)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
    # Initialize pipeline with configuration and command-line overrides
    pipeline = LegalDocumentPipeline(
        config=config,
        working_dir=args.working_dir if args.working_dir != 'data' else None,
        embedding_model=args.embedding_model if args.embedding_model != 'BAAI/bge-small-en-v1.5' else None,
        chunking_model=args.chunking_model if args.chunking_model != 'nlpaueb/legal-bert-base-uncased' else None,
        chunk_size=args.chunk_size if args.chunk_size != 384 else None,
        min_chunk_size=args.min_chunk_size if args.min_chunk_size != 100 else None,
        qdrant_url=args.qdrant_url if args.qdrant_url != 'http://localhost:6333' else None
    )
    
    if args.status:
        # Show status and exit
        status = pipeline.get_pipeline_status()
        print(json.dumps(status, indent=2))
        return
    
    # Run the pipeline
    try:
        results = pipeline.run_full_pipeline(
            court=args.court,
            num_dockets=args.num_dockets,
            batch_size=args.batch_size,
            create_vector=not args.no_vector,
            create_hybrid=args.enable_hybrid,
            vector_collection=args.vector_collection,
            hybrid_collection=args.hybrid_collection,
            job_id=args.job_id,
            resume=args.resume,
            skip_duplicates=not args.no_skip_duplicates,
            duplicate_check_mode=args.duplicate_check_mode,
            overwrite_collection=args.overwrite_collection
        )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📄 Documents: {results.get('raw_documents')}")
        print(f"🔢 Chunks: {results.get('chunks')}")
        if results.get('vector_collection'):
            print(f"🔍 Vector collection: {results['vector_collection']}")
        if results.get('hybrid_collection'):
            print(f"⚡ Hybrid collection: {results['hybrid_collection']}")
        
    except KeyboardInterrupt:
        logger.info("❌ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()