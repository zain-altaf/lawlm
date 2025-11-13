### fetches, processes, vectorizes, and uploads data to Qdrant
### can be combined with orchestration to trigger on a schedule

from dotenv import load_dotenv
import os
import logging
import urllib
import json
import uuid
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from datetime import datetime

from opinion_utills import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv('CASELAW_API_KEY')
headers = {'Authorization': f'Token {api_key}'} if api_key else {}

# Load config from project root (works both locally and in Docker)
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yml')
if not os.path.exists(config_path):
    # Fallback for Docker: config is mounted at /app/config.yml
    config_path = '/app/config.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

chunk_size_chars = config['chunking']['chunk_size_chars']
overlap_chars = config['chunking']['overlap_chars']
min_chunk_size_chars = config['chunking']['min_chunk_size_chars']
quality_threshold = config['chunking']['quality_threshold']

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size_chars,
    chunk_overlap=overlap_chars,
    length_function=len,
    separators=config['chunking']['separators']
)

embedding_model = config['vectorization']['embedding_model']
batch_size = config['vectorization']['batch_size']
memory_cleanup_frequency = config['vectorization']['memory_cleanup_frequency']
device = config['vectorization']['device']
vector_size = config['vectorization']['vector_size']
collection_name_vector = config['qdrant']['collection_name']

request_delay = config['api']['request_delay']
max_retries = config['api']['max_retries']
retry_delay = config['api']['retry_delay']
court = config['api']['court']

def fetch_docket_page(cursor, existing_docket_ids, court, page_count):
    """Fetch a single page of dockets from CourtListener.

    Args:
        cursor: Pagination cursor for CourtListener API
        existing_docket_ids: Set of existing docket IDs in Qdrant
        court: Court identifier (e.g., 'scotus')
        page_count: Current page number for logging

    Returns:
        tuple: (page_dockets, next_cursor) or (None, None) on error
    """
    base_url = "https://www.courtlistener.com/api/rest/v4/dockets/"

    try:
        params = {
            "court": court,
            "ordering": "id"  # Ordering by id for consistent pagination (earliest IDs = oldest cases)
        }
        if cursor:
            params["cursor"] = cursor

        # Construct full URL with params
        query_string = urllib.parse.urlencode(params)
        full_url = f"{base_url}?{query_string}"

        response_data = api_request_with_retry(full_url, headers, max_retries, retry_delay, request_delay)

        if response_data is None:
            logger.error(f"Failed to fetch page {page_count}")
            return None, None

        # Get dockets from this page
        page_dockets = response_data.get('results', [])

        # Get next cursor - extract just the cursor value, not the full URL
        next_url = response_data.get('next')
        next_cursor = None
        if next_url:
            parsed = urllib.parse.urlparse(next_url)
            query_params = urllib.parse.parse_qs(parsed.query)
            next_cursor = query_params.get('cursor', [None])[0]

        # Filter out existing dockets and add metadata
        filtered_dockets = []
        for docket in page_dockets:
            docket_id = docket.get('id', '')
            if not docket_id:
                raise ValueError(f"Docket without ID found on page {page_count} (cursor={cursor})")

            if docket_id in existing_docket_ids:
                logger.info(f"Skipping existing docket ID: {docket_id}")
                continue

            # Save the NEXT cursor (for resuming from the next page)
            docket["page_cursor"] = next_cursor
            filtered_dockets.append(docket)

        # Replace page_dockets with filtered list
        page_dockets = filtered_dockets

        logger.info(f"Page {page_count}: fetched {len(page_dockets)} new dockets (after filtering duplicates)")
        return page_dockets, next_cursor

    except Exception as e:
        logger.error(f"Unexpected error on page {page_count}: {e}")
        return None, None


def process_docket(docket, next_cursor):
    """Process a single docket to extract clusters and opinions.

    Args:
        docket: Docket dictionary from CourtListener API
        next_cursor: Cursor for the next page

    Returns:
        list: opinions extracted from this docket
    """
    opinions = []
    docket_id = docket.get('id', '')

    if not docket_id:
        raise ValueError(f"Docket without ID found with cursor: {docket['page_cursor']}")

    logger.debug(f"Processing docket {docket_id}")

    for cluster_url in docket.get('clusters', []):
        try:
            logger.debug(f"Processing cluster: {cluster_url}")

            cluster_data = api_request_with_retry(cluster_url, headers, max_retries, retry_delay, request_delay)

            if cluster_data is None:
                logger.error(f"Failed to fetch cluster {cluster_url}")
                continue

            cluster_id = cluster_data.get('id', '')

            for opinion_url in cluster_data.get("sub_opinions", []):
                try:
                    logger.debug(f"Processing opinion: {opinion_url}")

                    opinion_data = api_request_with_retry(opinion_url, headers, max_retries, retry_delay, request_delay)

                    if opinion_data is None:
                        logger.warning(f"Failed to fetch opinion {opinion_url}")
                        continue

                    # Extract text from available fields in order of recommendation
                    raw_text = None
                    source_field = None
                    for field in [
                        'html_with_citations',
                        'plain_text',
                        'html_columbia',
                        'html_lawbox',
                        'html_anon_2020',
                        'html'
                    ]:
                        if opinion_data.get(field):
                            raw_text = opinion_data[field]
                            source_field = field
                            break

                    if not raw_text or len(raw_text.strip()) < 100:
                        logger.debug(f"Skipping opinion {opinion_data.get('id')} - insufficient text")
                        continue

                    # Process text using existing enhanced processing
                    try:
                        processed = enhanced_text_processing(raw_text)

                        opinion_record = {
                            "docket_id": docket_id,
                            "cluster_id": cluster_id,
                            "opinion_id": opinion_data.get("id"),
                            "docket_number": docket.get("docket_number", "unknown"),
                            "case_name": cluster_data.get("case_name", "Unknown Case"),
                            "court_id": docket.get("court_id", "unknown"),
                            "judges": cluster_data.get("judges", ""),
                            "author": opinion_data.get("author_id", ""),
                            "opinion_type": opinion_data.get("type", "unknown"),
                            "date_filed": cluster_data.get("date_filed"),
                            "precedential_status": cluster_data.get("precedential_status"),
                            "sha1": opinion_data.get("sha1"),
                            "download_url": opinion_data.get("download_url"),
                            "source_field": source_field,
                            "opinion_text": processed['cleaned_text'],
                            "citations": processed['citations'],
                            "legal_entities": processed['legal_entities'],
                            "text_stats": processed['text_stats'],
                            "date_created": opinion_data.get("date_created"),
                            "date_modified": opinion_data.get("date_modified"),
                            "cursor": next_cursor,
                        }

                        opinions.append(opinion_record)
                        logger.debug(f"Successfully processed opinion {opinion_data.get('id')}")

                    except Exception as e:
                        logger.warning(f"Failed to process text for opinion {opinion_data.get('id')}: {e}")
                        continue

                except Exception as e:
                    logger.warning(f"Failed to fetch opinion {opinion_url}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to fetch cluster {cluster_url}: {e}")
            continue

    if opinions:
        logger.info(f"Docket {docket_id}: extracted {len(opinions)} opinions")

    return opinions


def get_chunks(docket_opinions):
    """Split opinions into semantic chunks.

    Args:
        docket_opinions: List of opinion dictionaries with text content

    Returns:
        list: List of chunk dictionaries with metadata
    """
    all_chunks = []

    for op in docket_opinions:
        opinion_text = op.get('opinion_text', '')
        if not opinion_text or len(opinion_text.strip()) < 50:
            continue

        raw_chunks = text_splitter.split_text(opinion_text)
        
        text_chunks = fix_chunk_overlaps(raw_chunks, min_chunk_size_chars)
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < min_chunk_size_chars:
                continue

            docket_id = op.get('docket_id', '')
            opinion_id = op.get('opinion_id', 'unknown')
            chunk = {
                "id": f"{docket_id}_{opinion_id}_{chunk_idx}",
                "docket_id": op.get('docket_id', ''),
                "cluster_id": op.get('cluster_id', ''),
                "opinion_id": op.get('opinion_id', ''),
                "chunk_id": f"{op.get('opinion_id', 'unknown')}_{chunk_idx}",
                "docket_number": op.get('docket_number', ''),
                "case_name": op.get('case_name', ''),
                "court_id": op.get('court_id', ''),
                "chunk_index": chunk_idx,
                "judges": op.get('judges', ''),
                "author": op.get('author', ''),
                "opinion_type": op.get('opinion_type', ''),
                "date_filed": op.get('date_filed', ''),
                "text": chunk_text.strip(),
                "precedential_status": op.get('precedential_status', ''),
                "sha1": op.get('sha1', ''),
                "download_url": op.get('download_url', ''),
                "source_field": op.get('source_field', ''),
                "citations": op.get('citations', []),
                "legal_entities": op.get('legal_entities', {}),
                "date_created": op.get('date_created', ''),
                "date_modified": op.get('date_modified', ''),
                "cursor": op.get('cursor', '')
            }

            all_chunks.append(chunk)
    
    return all_chunks


def vectorize_chunks(chunks, embedder, enhanced_text_processor):
    """Create vector embeddings for text chunks.

    Args:
        chunks: List of chunk dictionaries with text content
        embedder: SentenceTransformer model for creating embeddings
        enhanced_text_processor: Function to process and extract legal entities

    Returns:
        tuple: (embeddings array, enhanced_texts list) or None if no chunks
    """
    if not chunks:
        return None

    enhanced_texts = []
    for chunk in chunks:
        processed = enhanced_text_processor(chunk['text'])
        enhanced_text = processed['cleaned_text']
        chunk['extracted_citations'] = processed['citations']
        chunk['extracted_entities'] = processed['legal_entities']
        enhanced_texts.append(enhanced_text)

    embeddings = embedder.encode(
        enhanced_texts,
        batch_size=min(len(enhanced_texts), 16),
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings, enhanced_texts


def upload_to_qdrant(chunks, embeddings, enhanced_texts, qdrant_client, collection_name, vector_size):
    """Upload vectorized chunks to Qdrant database.

    Args:
        chunks: List of chunk dictionaries with metadata
        embeddings: Numpy array of dense vector embeddings
        enhanced_texts: List of processed text strings for BM25
        qdrant_client: QdrantClient instance
        collection_name: Name of Qdrant collection to upload to
        vector_size: Dimension of dense vectors

    Returns:
        dict: Status dictionary with vectors_created count or error
    """
    if not chunks or embeddings is None:
        return {'status': 'no_chunks', 'vectors_created': 0}

    try:
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())

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
                'date_modified': chunk.get('date_modified', ''),
                'time_processed': datetime.now().strftime("%d-%m-%y %H:%M:%S"),
                'cursor': chunk.get('cursor', '')
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

        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        return {
            'status': 'success',
            'vectors_created': len(points),
            'collection': collection_name
        }

    except Exception as e:
        logger.error(f"Error uploading to Qdrant: {e}")
        return {
            'status': 'error',
            'vectors_created': 0,
            'error': str(e)
        }

def ingestion(num_pages=1, court=court):
    """Main ingestion pipeline that processes data incrementally.

    For each page:
      For each docket in page:
        1. Process docket to extract opinions
        2. Chunk opinions
        3. Vectorize chunks
        4. Upload to Qdrant
      Move to next docket

    Args:
        num_pages: Number of pages to process
        court: Court identifier
    """

    cursor = None
    page_count = 0
    total_dockets = 0
    total_opinions = 0
    total_chunks = 0
    total_vectors = 0
    processed_dockets_this_run = set()  # Track dockets processed in this run

    # ensure you are connected to the qdrant client
    try:
        qdrant_client = get_qdrant_client()
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to get Qdrant client: {e}")
        return

    # ensures collection exists
    get_qdrant_collection(qdrant_client, collection_name_vector, vector_size)

    # fetch existing docket_ids and cursor for page navigation
    existing_docket_ids, cursor = get_existing_ids_and_cursor(qdrant_client, collection_name_vector)
    
    logger.info(f"Found {len(existing_docket_ids)} existing docket IDs in Qdrant")

    # get the embedding model instantiated
    logger.info(f"Loading embedding model: {embedding_model}")
    embedder = SentenceTransformer(embedding_model)
    embedder.eval()


    logger.info(f"Starting ingestion pipeline: {num_pages} pages from court: {court}")

    while page_count < num_pages:
        page_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Page {page_count}/{num_pages}")
        logger.info(f"{'='*60}")

        page_dockets, next_cursor = fetch_docket_page(cursor, existing_docket_ids, court, page_count)

        if page_dockets is None:
            logger.error(f"Failed to fetch page {page_count}, stopping pipeline")
            break

        logger.info(f"Fetched {len(page_dockets)} dockets on this page")

        for docket_idx, docket in enumerate(page_dockets, 1):
            docket_id = docket.get('id', 'unknown')

            logger.info(f"\n  Docket {docket_idx}/{len(page_dockets)} (ID: {docket_id})")

            docket_opinions = process_docket(docket, next_cursor)

            if not docket_opinions:
                logger.info(f"No opinions found, skipping docket")
                continue

            logger.info(f"Extracted {len(docket_opinions)} opinions")
            total_opinions += len(docket_opinions)

            docket_chunks = get_chunks(docket_opinions)
            logger.info(f"Created {len(docket_chunks)} chunks")
            total_chunks += len(docket_chunks)

            if not docket_chunks:
                logger.warning(f"No chunks created, skipping docket")
                continue

            logger.info(f"Vectorizing {len(docket_chunks)} chunks...")
            embeddings, enhanced_texts = vectorize_chunks(docket_chunks, embedder, enhanced_text_processing)

            logger.info(f"Uploading {len(docket_chunks)} chunks to Qdrant...")
            result = upload_to_qdrant(docket_chunks, embeddings, enhanced_texts, qdrant_client, collection_name_vector, vector_size)

            if result['status'] == 'success':
                total_vectors += result['vectors_created']
                processed_dockets_this_run.add(docket_id)  # Mark as processed
                logger.info(f"Docket {docket_id} complete!")
                total_dockets += 1
            else:
                logger.error(f"Failed to upload docket {docket_id}: {result.get('error')}")

        logger.info(f"\nPage {page_count} complete: processed {len(page_dockets)} dockets")

        if not next_cursor:
            logger.info(f"Reached end of available pages")
            break

        cursor = next_cursor

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion Pipeline Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"   Pages processed: {page_count}")
    logger.info(f"   Dockets processed: {total_dockets}")
    logger.info(f"   Total opinions: {total_opinions}")
    logger.info(f"   Total chunks: {total_chunks}")
    logger.info(f"   Total vectors: {total_vectors}")
    logger.info(f"{'='*60}\n")

def main(num_pages=1, court=court):
    ingestion(num_pages=num_pages, court=court)


if __name__ == "__main__":
    main(num_pages=1, court=court)