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
import requests
import os
from dotenv import load_dotenv

# Import our processing modules
from fetch_and_process import process_docket
from config import PipelineConfig, load_config
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

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and available data."""
        status = {
            'working_dir': str(self.working_dir),
            'files': [str(f) for f in self.working_dir.glob("*")],
            'configuration': self.config.dict()
        }
        return status

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the legal document processing pipeline.
        Process each docket completely before moving to next (incremental processing).
        Includes smart pagination and deduplication.
        """
        logger.info(f"🚀 Starting legal document processing pipeline")
        logger.info(f"🏛️ Court: {self.config.data_ingestion.court}, Dockets: {self.config.data_ingestion.num_dockets}")
        start_time = datetime.now()
        
        # Initialize vector processor with deduplication support
        vector_processor = None
        if VECTOR_PROCESSOR_AVAILABLE and self.config.vector_processing.collection_name_vector:
            vector_processor = EnhancedVectorProcessor(
                model_name=self.config.vector_processing.embedding_model,
                collection_name=self.config.vector_processing.collection_name_vector,
                qdrant_url=self.config.qdrant.url
            )
            logger.info(f"🤖 Vector processor initialized with deduplication")

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.text_splitter.chunk_size_chars,
            chunk_overlap=self.config.text_splitter.overlap_chars,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )
        
        # Get all unique dockets in Qdrant
        existing_dockets = set()
        if vector_processor:
            existing_dockets = vector_processor.get_existing_docket_numbers()
            logger.info(f"🔍 Found {len(existing_dockets)} existing dockets in collection")
        
        # Stats tracking
        stats = {
            'total_requested': self.config.data_ingestion.num_dockets,
            'dockets_fetched': 0,
            'dockets_processed': 0,
            'documents_processed': 0,
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

            stats['dockets_fetched'] = len(new_dockets)  # These are all new
            stats['duplicates_skipped'] = 0  # Will be calculated during smart pagination
            
            logger.info(f"📊 Smart pagination found {len(new_dockets)} new dockets to process")
            
            if not new_dockets:
                logger.info("✅ No new dockets to process - collection is up to date")
                return {
                    'status': 'up_to_date',
                    'stats': stats,
                    'message': 'All requested dockets already exist in collection'
                }
            
            # Step 2: Process each NEW docket incrementally  
            for idx, docket in enumerate(new_dockets, 1):
                docket_number = docket.get('docket_number', '')
                if not docket_number:
                    continue
                    
                logger.info(f"\n[{idx}/{len(new_dockets)}] Processing docket {docket_number}")
                
                try:
                    # Fetch documents for this specific docket directly
                    docket_docs = self._fetch_docket_documents(docket, self.config.data_ingestion.court)
                    stats['documents_processed'] += len(docket_docs)
                    
                    if not docket_docs:
                        logger.warning(f"⚠️ No documents found for docket {docket_number}")
                        continue
                    
                    # Chunk documents
                    all_chunks = []
                    for doc_idx, doc in enumerate(docket_docs):
                        opinion_text = doc.get('opinion_text', '')
                        if not opinion_text or len(opinion_text.strip()) < 50:
                            continue

                        text_chunks = text_splitter.split_text(opinion_text)
                        
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            if len(chunk_text.strip()) < self.config.text_splitter.min_chunk_size_chars:
                                continue

                            chunk = {
                                "chunk_id": f"{doc['id']}_{chunk_idx}",
                                "document_id": str(doc.get('id', f'doc_{doc_idx}')),
                                "docket_number": doc.get('docket_number', ''),
                                "case_name": doc.get('case_name', ''),
                                "court_id": doc.get('court_id', ''),
                                "chunk_index": chunk_idx,
                                "text": chunk_text.strip(),
                                "token_count": len(chunk_text.split()),
                                "citation_count": len(re.findall(r'\d+\s+[A-Z][a-z\.]*\s+\d+', chunk_text)),
                                "author": doc.get('author', ''),
                                "opinion_type": doc.get('opinion_type', ''),
                                "date_filed": doc.get('date_filed', '')
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
                                result = vector_processor.process_and_upload_batch(
                                    batch,
                                    collection_name=self.config.vector_processing.collection_name_vector
                                )
                                stats['vectors_uploaded'] += result.get('vectors_created', 0)
                            except Exception as e:
                                logger.error(f"Failed to upload batch: {e}")
                                stats['errors'].append({'docket': docket_number, 'error': str(e)})
                    
                    stats['dockets_processed'] += 1
                    logger.info(f"✅ Docket {docket_number}: {len(docket_docs)} docs, {len(all_chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing docket {docket_number}: {e}")
                    stats['errors'].append({'docket': docket_number, 'error': str(e)})
            
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
            logger.info(f"📄 Documents: {stats['documents_processed']}")
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
        Fetch dockets using linear pagination, checking each docket against existing ones.
        More reliable than calculated page skipping since API ordering may not be guaranteed.
        """
        
        load_dotenv()
        CASELAW_API_KEY = os.getenv('CASELAW_API_KEY')
        HEADERS = {'Authorization': f'Token {CASELAW_API_KEY}'} if CASELAW_API_KEY else {}
        
        new_dockets = []
        page_count = 0
        page_size = 200  # Use larger page size to reduce API calls
        consecutive_empty_pages = 0
        max_consecutive_empty = 50  # Keep going until we find unprocessed older dockets

        logger.info(f"🌐 Cursor-based pagination: fetching {num_dockets} NEW dockets (oldest first)...")
        logger.info(f"🔍 Found {len(existing_dockets)} existing dockets in collection")
        
        base_url = "https://www.courtlistener.com/api/rest/v4/dockets/"
        
        # Start with cursor-based pagination to access older dockets
        cursor = None
        
        while len(new_dockets) < num_dockets:
            page_count += 1
            
            logger.info(f"📄 Fetching page {page_count} - need {num_dockets - len(new_dockets)} more NEW dockets")
            
            try:
                params = {
                    "court": court,
                    "ordering": "id"  # Oldest dockets first (to access unprocessed historical data)
                }
                if cursor:
                    params["cursor"] = cursor
                
                response = requests.get(
                    base_url,
                    params=params,
                    headers=HEADERS,
                    timeout=30
                )
                response.raise_for_status()
                
                response_data = response.json()
                page_dockets = response_data.get('results', [])
                if not page_dockets:
                    logger.warning(f"⚠️ No more dockets available from API")
                    break
                
                # Get next cursor for continuation
                next_url = response_data.get('next')
                if next_url:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(next_url)
                    query_params = urllib.parse.parse_qs(parsed.query)
                    cursor = query_params.get('cursor', [None])[0]
                else:
                    cursor = None
                
                # Check this page for new dockets
                page_new_count = 0
                for docket in page_dockets:
                    docket_number = docket.get('docket_number', '')
                    if docket_number and docket_number not in existing_dockets:
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
                
            except requests.RequestException as e:
                logger.error(f"❌ API error on page {page_count}: {e}")
                break
        
        final_new_dockets = new_dockets[:num_dockets]  # Limit to exactly what was requested
        
        logger.info(f"🎯 Cursor-based pagination complete:")
        logger.info(f"   📄 Pages fetched: {page_count}")
        logger.info(f"   ✨ New dockets found: {len(final_new_dockets)}")
        
        return final_new_dockets

    def _fetch_docket_documents(self, docket: Dict[str, Any], court: str) -> List[Dict[str, Any]]:
        """Fetch and process documents for a specific docket object."""
        import requests
        import os
        from dotenv import load_dotenv
        from fetch_and_process import enhanced_text_processing
        
        load_dotenv()
        CASELAW_API_KEY = os.getenv('CASELAW_API_KEY')
        HEADERS = {'Authorization': f'Token {CASELAW_API_KEY}'} if CASELAW_API_KEY else {}
        
        documents = []
        docket_number = docket.get('docket_number', '')
        
        logger.info(f"📥 Fetching documents for docket {docket_number}")
        
        # Process clusters and opinions from the docket
        for cluster_url in docket.get("clusters", []):
            try:
                logger.debug(f"Processing cluster: {cluster_url}")
                cluster_resp = requests.get(cluster_url, headers=HEADERS, timeout=30)
                cluster_resp.raise_for_status()
                cluster = cluster_resp.json()
                
                for opinion_url in cluster.get("sub_opinions", []):
                    try:
                        logger.debug(f"Processing opinion: {opinion_url}")
                        opinion_resp = requests.get(opinion_url, headers=HEADERS, timeout=30)
                        opinion_resp.raise_for_status()
                        opinion = opinion_resp.json()
                        
                        # Extract text from available fields (in priority order)
                        raw_text = None
                        source_field = None
                        for field in ['html_columbia', 'html_lawbox', 'html_anon_2020', 
                                     'html_with_citations', 'html', 'plain_text']:
                            if opinion.get(field):
                                raw_text = opinion[field]
                                source_field = field
                                break
                        
                        if not raw_text or len(raw_text.strip()) < 100:
                            logger.debug(f"Skipping opinion {opinion.get('id')} - insufficient text")
                            continue
                        
                        # Process text using existing enhanced processing
                        try:
                            processed = enhanced_text_processing(raw_text)
                            
                            document = {
                                "id": opinion.get("id"),
                                "docket_number": docket_number,
                                "case_name": cluster.get("case_name", "Unknown Case"),
                                "court_id": court,
                                "judges": cluster.get("judges", ""),
                                "author": opinion.get("author_str", ""),
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
                                "processing_timestamp": datetime.now().isoformat(),
                                "ready_for_chunking": True
                            }
                            
                            documents.append(document)
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
        
        logger.info(f"📄 Found {len(documents)} documents in docket {docket_number}")
        return documents

def main():
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
        print(json.dumps(status, indent=2))
        return

    # Run pipeline
    results = pipeline.run_pipeline()
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()