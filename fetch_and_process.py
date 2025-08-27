import requests
import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from batch_utils import BatchProcessor, create_job_id, merge_json_files

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
CASELAW_API_KEY = os.getenv("CASELAW_API_KEY")

# --- API Headers ---
HEADERS = {"Authorization": f"Token {CASELAW_API_KEY}"}

# --- Data Fetching and Processing ---

def extract_legal_citations(text: str) -> List[str]:
    """Extract legal citations from text using regex patterns."""
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
    
    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)
    
    return list(set(citations))  # Remove duplicates

def extract_legal_entities(text: str) -> Dict[str, List[str]]:
    """Extract legal entities like judges, parties, courts, etc."""
    entities = {
        'judges': [],
        'parties': [],
        'courts': [],
        'statutes': []
    }
    
    # Judge patterns: "Justice [Name]", "Judge [Name]", "Chief Justice [Name]"
    judge_patterns = [
        r'(?:Justice|Judge|Chief Justice|Associate Justice)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'([A-Z][a-z]+),?\s+(?:J\.|C\.J\.|Associate Justice|Chief Justice)'
    ]
    
    for pattern in judge_patterns:
        matches = re.findall(pattern, text)
        entities['judges'].extend([match.strip() for match in matches if isinstance(match, str)])
    
    # Party patterns: "Plaintiff v. Defendant" format
    party_pattern = r'([A-Z][a-zA-Z\s&,\.]+?)\s+v\.?\s+([A-Z][a-zA-Z\s&,\.]+?)(?:\s|,|\.|\n)'
    party_matches = re.findall(party_pattern, text)
    for plaintiff, defendant in party_matches:
        entities['parties'].extend([plaintiff.strip(), defendant.strip()])
    
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
        entities['courts'].extend(matches)
    
    # Statute patterns: "42 U.S.C. § 1983", "Title VII", etc.
    statute_patterns = [
        r'\b\d+\s+U\.S\.C\.?\s*§+\s*\d+[a-z]*(?:\([^)]+\))*',
        r'Title\s+[IVX]+(?:\s+of\s+[^,.\n]+)?',
        r'Section\s+\d+[a-z]*(?:\([^)]+\))*'
    ]
    
    for pattern in statute_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['statutes'].extend(matches)
    
    # Clean and deduplicate
    for key in entities:
        entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
    
    return entities

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
    citations = extract_legal_citations(cleaned)
    entities = extract_legal_entities(cleaned)
    
    return {
        'cleaned_text': cleaned,
        'citations': citations,
        'legal_entities': entities,
        'text_stats': {
            'length': len(cleaned),
            'word_count': len(cleaned.split()),
            'citation_count': len(citations)
        }
    }

def process_docket(court: str = 'scotus', num_dockets: int = 5, progress_callback=None) -> List[Dict[str, Any]]:
    """
    Fetches dockets, clusters, and opinions from CourtListener API,
    with enhanced legal text processing for hybrid search pipeline.
    
    Args:
        court: Court identifier (e.g., 'scotus', 'ca1', 'ca2')
        num_dockets: Number of dockets to fetch
        progress_callback: Optional callback function for progress updates
    
    Returns:
        List of processed document dictionaries
    
    Raises:
        ValueError: If invalid parameters provided
        requests.RequestException: If API requests fail
        RuntimeError: If no documents could be processed
    """
    # Validate inputs
    if num_dockets < 1:
        raise ValueError(f"num_dockets must be at least 1, got {num_dockets}")
    
    if not CASELAW_API_KEY:
        raise RuntimeError("CASELAW_API_KEY environment variable is required")
    
    print(f"📥 Fetching {num_dockets} dockets from court: {court}...")
    print(f"🔍 Enhanced processing: extracting citations, legal entities, and metadata...")
    
    if progress_callback:
        progress_callback('started', {'court': court, 'num_dockets': num_dockets})
    
    try:
        # CourtListener API has pagination limits, so we may need multiple requests
        all_dockets = []
        page = 1
        max_page_size = min(num_dockets, 200)  # API limit is typically 200
        
        while len(all_dockets) < num_dockets:
            remaining_needed = num_dockets - len(all_dockets)
            page_size = min(remaining_needed, max_page_size)
            
            docket_resp = requests.get(
                "https://www.courtlistener.com/api/rest/v4/dockets/",
                params={
                    "court": court, 
                    "page_size": page_size,
                    "page": page
                },
                headers=HEADERS,
                timeout=30
            )

            docket_resp.raise_for_status()
            
            page_dockets = docket_resp.json().get("results", [])
            if not page_dockets:
                print(f"⚠️ No more dockets available (got {len(all_dockets)} total)")
                break
                
            all_dockets.extend(page_dockets)
            print(f"📄 Page {page}: Retrieved {len(page_dockets)} dockets (total: {len(all_dockets)})")
            
            # Stop if we got fewer results than requested (end of available dockets)
            if len(page_dockets) < page_size:
                break
                
            page += 1
            
        dockets = all_dockets[:num_dockets]  # Limit to exactly what was requested
        
        if not dockets:
            raise RuntimeError(f"No dockets found for court '{court}'. Check court identifier.")
            
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request timeout when fetching dockets for court '{court}'")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch dockets for court '{court}': {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching dockets: {e}")

    all_data = []
    processed_count = 0
    skipped_count = 0
    total_dockets = len(dockets)
    api_errors = 0
    processing_errors = 0
    
    print(f"📊 Processing {total_dockets} dockets...")
    
    for docket_idx, docket in enumerate(dockets):
        docket_number = docket.get("docket_number", "Unknown")
        print(f"[{docket_idx + 1}/{total_dockets}] Processing docket: {docket_number}")
        
        if progress_callback:
            progress_callback('docket_started', {
                'docket_number': docket_number,
                'progress': (docket_idx + 1) / total_dockets
            })
        
        clusters = docket.get("clusters", [])
        if not clusters:
            print(f"  ⚠️ No clusters found for docket {docket_number}")
            skipped_count += 1
            continue
            
        for cluster_idx, cluster_url in enumerate(clusters):
            try:
                cluster_resp = requests.get(
                    cluster_url, 
                    headers=HEADERS,
                    timeout=30
                )
                cluster_resp.raise_for_status()
                cluster = cluster_resp.json()
                
                print(f"  📄 Processing cluster {cluster_idx + 1}/{len(clusters)}")
                
                opinions = cluster.get("sub_opinions", [])
                if not opinions:
                    print(f"    ⚠️ No opinions found in cluster")
                    continue
                    
                for opinion_idx, opinion_url in enumerate(opinions):
                    try:
                        opinion_resp = requests.get(
                            opinion_url, 
                            headers=HEADERS,
                            timeout=30
                        )
                        opinion_resp.raise_for_status()
                        opinion = opinion_resp.json()
                        
                        print(f"    📝 Processing opinion {opinion_idx + 1}/{len(opinions)}")
                        
                        # Extract raw text from best available source
                        raw_text, source_field = "", 'Unknown'
                        for field in ['html_with_citations', 'html_columbia', 'html_lawbox', 'xml_harvard', 'html', 'plain_text']:
                            if opinion.get(field):
                                raw_text = opinion[field]
                                source_field = field
                                break
                        
                        if not raw_text or len(raw_text.strip()) < 100:
                            print(f"      ⚠️ Skipping opinion {opinion.get('id')} - insufficient text content")
                            skipped_count += 1
                            continue
                        
                        # Enhanced text processing
                        processed_text = enhanced_text_processing(raw_text)
                        
                        # Skip if processed text is too short
                        if processed_text['text_stats']['word_count'] < 50:
                            print(f"      ⚠️ Skipping opinion {opinion.get('id')} - text too short after processing")
                            skipped_count += 1
                            continue
                        
                        # Build comprehensive document record
                        document = {
                            # Core identifiers
                            "id": opinion.get("id"),
                            "docket_number": docket_number,
                            "case_name": cluster.get("case_name", "Unknown Case"),
                            "court_id": docket.get("court_id", "unknown"),
                            
                            # Legal metadata
                            "judges": cluster.get("judges", ""),
                            "author": opinion.get("author_str", ""),
                            "opinion_type": opinion.get("type", "unknown"),
                            "date_filed": cluster.get("date_filed"),
                            "precedential_status": cluster.get("precedential_status"),
                            
                            # Document properties
                            "sha1": opinion.get("sha1"),
                            "download_url": opinion.get("download_url"),
                            "source_field": source_field,
                            
                            # Enhanced text content
                            "opinion_text": processed_text['cleaned_text'],
                            "citations": processed_text['citations'],
                            "legal_entities": processed_text['legal_entities'],
                            "text_stats": processed_text['text_stats'],
                            
                            # Processing metadata
                            "processing_timestamp": None,  # Will be set during chunk processing
                            "ready_for_chunking": True
                        }
                        
                        all_data.append(document)
                        processed_count += 1
                        
                        if processed_count % 5 == 0:
                            print(f"    ✅ Processed {processed_count} opinions so far...")
                            
                        if progress_callback:
                            progress_callback('opinion_processed', {
                                'processed_count': processed_count,
                                'opinion_id': opinion.get('id')
                            })
                        
                    except requests.exceptions.Timeout:
                        print(f"      ❌ Timeout processing opinion {opinion_url}")
                        api_errors += 1
                        skipped_count += 1
                        continue
                    except requests.exceptions.RequestException as e:
                        print(f"      ❌ API error processing opinion {opinion_url}: {e}")
                        api_errors += 1
                        skipped_count += 1
                        continue
                    except Exception as e:
                        print(f"      ❌ Processing error for opinion {opinion_url}: {e}")
                        processing_errors += 1
                        skipped_count += 1
                        continue
                        
            except requests.exceptions.Timeout:
                print(f"    ❌ Timeout processing cluster {cluster_url}")
                api_errors += 1
                continue
            except requests.exceptions.RequestException as e:
                print(f"    ❌ API error processing cluster {cluster_url}: {e}")
                api_errors += 1
                continue
            except Exception as e:
                print(f"    ❌ Processing error for cluster {cluster_url}: {e}")
                processing_errors += 1
                continue
    
    # Final validation
    if not all_data:
        raise RuntimeError(f"No documents were successfully processed from {total_dockets} dockets")
    
    print(f"\n✅ Processing complete:")
    print(f"  📊 Dockets processed: {total_dockets}")
    print(f"  ✅ Successfully processed: {processed_count} opinions")
    print(f"  ⚠️ Skipped: {skipped_count} opinions")
    print(f"  🌐 API errors: {api_errors}")
    print(f"  ⚙️ Processing errors: {processing_errors}")
    print(f"  📄 Ready for semantic chunking: {len(all_data)} documents")
    
    if progress_callback:
        progress_callback('completed', {
            'total_documents': len(all_data),
            'processed_count': processed_count,
            'skipped_count': skipped_count,
            'api_errors': api_errors,
            'processing_errors': processing_errors
        })
    
    return all_data


def process_docket_in_batches(court: str = 'scotus', 
                             num_dockets: int = 5, 
                             batch_size: int = 5,
                             working_dir: str = "data",
                             job_id: Optional[str] = None,
                             resume: bool = False,
                             progress_callback=None) -> Dict[str, Any]:
    """
    Process dockets in batches with resume capability and progress tracking.
    
    Args:
        court: Court identifier
        num_dockets: Total number of dockets to process
        batch_size: Number of dockets per batch
        working_dir: Directory for intermediate files
        job_id: Optional job identifier (auto-generated if not provided)
        resume: Whether to resume from previous incomplete run
        progress_callback: Callback for progress updates
    
    Returns:
        Dictionary with processing results and file paths
    """
    from pathlib import Path
    from datetime import datetime
    
    # Initialize batch processor
    batch_processor = BatchProcessor(working_dir)
    
    # Create or validate job ID
    if job_id is None:
        job_id = create_job_id(court, num_dockets)
    
    print(f"🚀 Starting batch processing job: {job_id}")
    print(f"📊 Processing {num_dockets} dockets from {court} in batches of {batch_size}")
    
    # Handle resume logic
    if resume:
        try:
            batches, metadata = batch_processor.load_batch_state(job_id)
            print(f"📂 Resuming job '{job_id}' with {len(batches)} batches")
            
            # Validate that parameters match
            if (metadata.get('court') != court or 
                metadata.get('num_dockets') != num_dockets or
                metadata.get('batch_size') != batch_size):
                print("⚠️ Warning: Resume parameters don't match original job")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Cannot resume: {e}")
            print("🔄 Starting new job instead")
            resume = False
    
    # Create new batch plan if not resuming
    if not resume:
        try:
            batches = batch_processor.calculate_batches(num_dockets, batch_size)
            metadata = {
                'court': court,
                'num_dockets': num_dockets,
                'batch_size': batch_size,
                'working_dir': working_dir,
                'started_at': datetime.now().isoformat()
            }
            batch_processor.save_batch_state(job_id, batches, metadata)
        except ValueError as e:
            raise ValueError(f"Invalid batch parameters: {e}")
    
    # Get pending batches
    pending_batches = batch_processor.get_pending_batches(job_id)
    if not pending_batches:
        print("✅ All batches already completed!")
        return _finalize_batch_job(batch_processor, job_id, working_dir)
    
    print(f"📋 Processing {len(pending_batches)} pending batches")
    
    # Process each pending batch
    batch_files = []
    total_documents = 0
    
    for batch in pending_batches:
        # Get total batches count for progress display
        _, metadata = batch_processor.load_batch_state(job_id)
        total_batches = batch_processor.get_batch_progress(job_id)['total_batches']
        print(f"\n📦 Processing batch {batch.batch_id + 1}/{total_batches} ({batch.size} dockets)")
        
        # Update batch status
        batch_processor.update_batch_status(job_id, batch.batch_id, "processing")
        
        try:
            # Process this batch of dockets
            batch_documents = process_docket(
                court=court, 
                num_dockets=batch.size,
                progress_callback=progress_callback
            )
            
            if not batch_documents:
                error_msg = f"No documents processed in batch {batch.batch_id}"
                print(f"❌ {error_msg}")
                batch_processor.update_batch_status(job_id, batch.batch_id, "failed", error_msg)
                continue
            
            # Save batch results
            batch_filename = f"batch_{job_id}_{batch.batch_id:03d}.json"
            batch_filepath = Path(working_dir) / batch_filename
            
            with open(batch_filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_documents, f, ensure_ascii=False, indent=2)
            
            batch_files.append(str(batch_filepath))
            total_documents += len(batch_documents)
            
            # Update batch status
            batch_processor.update_batch_status(job_id, batch.batch_id, "completed")
            
            print(f"✅ Batch {batch.batch_id + 1} completed: {len(batch_documents)} documents")
            print(f"💾 Saved to: {batch_filepath}")
            
            # Progress callback
            if progress_callback:
                progress_callback('batch_completed', {
                    'batch_id': batch.batch_id,
                    'documents_in_batch': len(batch_documents),
                    'total_documents_so_far': total_documents
                })
        
        except Exception as e:
            error_msg = f"Error processing batch {batch.batch_id}: {str(e)}"
            print(f"❌ {error_msg}")
            batch_processor.update_batch_status(job_id, batch.batch_id, "failed", error_msg)
            continue
    
    # Finalize job
    return _finalize_batch_job(batch_processor, job_id, working_dir)


def _finalize_batch_job(batch_processor: BatchProcessor, job_id: str, working_dir: str) -> Dict[str, Any]:
    """
    Finalize batch processing job by merging results and cleaning up.
    
    Args:
        batch_processor: Batch processor instance
        job_id: Job identifier
        working_dir: Working directory
        
    Returns:
        Final processing results
    """
    from pathlib import Path
    
    print(f"\n🔄 Finalizing batch job: {job_id}")
    
    # Get job progress
    progress = batch_processor.get_batch_progress(job_id)
    
    if progress.get('completed', 0) == 0:
        print("❌ No batches completed successfully")
        return {
            'success': False,
            'error': 'No batches completed successfully',
            'progress': progress
        }
    
    # Find all batch files
    working_path = Path(working_dir)
    batch_files = list(working_path.glob(f"batch_{job_id}_*.json"))
    batch_files.sort()  # Ensure proper order
    
    if not batch_files:
        print("❌ No batch files found for merging")
        return {
            'success': False,
            'error': 'No batch files found',
            'progress': progress
        }
    
    # Merge batch files
    final_filename = f"final_{job_id}.json"
    final_filepath = working_path / final_filename
    
    print(f"📋 Merging {len(batch_files)} batch files...")
    merge_stats = merge_json_files(
        [str(f) for f in batch_files], 
        str(final_filepath)
    )
    
    # Clean up intermediate files
    batch_processor.cleanup_batch_files(job_id, keep_final=True)
    
    # Final results
    results = {
        'success': True,
        'job_id': job_id,
        'final_file': str(final_filepath),
        'total_documents': merge_stats['total_items'],
        'batches_processed': merge_stats['files_processed'],
        'batches_failed': progress.get('failed', 0),
        'completion_percentage': progress.get('completion_percentage', 0),
        'merge_stats': merge_stats,
        'progress': progress
    }
    
    print(f"\n🎉 Batch processing completed!")
    print(f"📄 Total documents: {results['total_documents']}")
    print(f"✅ Successful batches: {results['batches_processed']}")
    print(f"❌ Failed batches: {results['batches_failed']}")
    print(f"💾 Final file: {final_filepath}")
    
    return results


def main():
    """Main function to run the enhanced data ingestion process for hybrid search."""
    print("🏛️  Legal Document Ingestion Pipeline")
    print("=" * 50)
    print("Enhanced processing with citation extraction and legal entity recognition")
    print("Optimized for hybrid search with semantic chunking")
    
    # Get command line arguments for court and number of dockets
    import argparse
    parser = argparse.ArgumentParser(description='Fetch and process legal case data for hybrid search')
    parser.add_argument('--court', default='scotus', help='Court identifier (default: scotus)')
    parser.add_argument('--num_dockets', type=int, default=5, help='Number of dockets to fetch (default: 5)')
    parser.add_argument('--output', default='data/raw_cases_enhanced.json', help='Output filename (default: data/raw_cases_enhanced.json)')
    parser.add_argument('--min_word_count', type=int, default=50, help='Minimum word count for opinion text (default: 50)')
    args = parser.parse_args()

    # Validate inputs
    if args.num_dockets < 1:
        print("Error: num_dockets must be at least 1")
        return
    if args.num_dockets > 100:
        print("Warning: Large numbers of dockets may take significant time and hit API rate limits")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def progress_callback(stage, data):
        """Progress callback for tracking processing stages."""
        if stage == 'started':
            print(f"🚀 Starting ingestion for {data['court']} court")
        elif stage == 'docket_started':
            progress_pct = int(data['progress'] * 100)
            print(f"📁 [{progress_pct}%] Processing docket: {data['docket_number']}")
        elif stage == 'opinion_processed':
            if data['processed_count'] % 10 == 0:
                print(f"📝 Processed {data['processed_count']} opinions...")
        elif stage == 'completed':
            print(f"🎉 Ingestion completed: {data['total_documents']} documents ready")
    
    try:
        # Fetch and process documents with enhanced legal text processing
        docs = process_docket(
            court=args.court, 
            num_dockets=args.num_dockets,
            progress_callback=progress_callback
        )
        
        if not docs:
            print("❌ No documents were successfully processed. Check your API key and court identifier.")
            print("💡 Common issues:")
            print("   - Invalid CASELAW_API_KEY environment variable")
            print("   - Invalid court identifier (try 'scotus', 'ca1', 'ca2', etc.)")
            print("   - Network connectivity issues")
            return
        
        # Generate processing summary
        total_citations = sum(len(doc['citations']) for doc in docs)
        total_words = sum(doc['text_stats']['word_count'] for doc in docs)
        
        # Save enhanced documents to JSON file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Enhanced data ingestion completed successfully!")
        print(f"📄 Documents processed: {len(docs)}")
        print(f"📚 Total citations found: {total_citations}")
        print(f"📝 Total words processed: {total_words:,}")
        print(f"💾 Data saved to: {args.output}")
        print(f"\n🔄 Next step: Run complete pipeline")
        print("   python pipeline_runner.py --court scotus --num-dockets 5")
        
    except Exception as e:
        print(f"❌ Error during data ingestion: {e}")
        print("Please check your CourtListener API key and internet connection.")
        raise

if __name__ == "__main__":
    main()