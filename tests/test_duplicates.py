"""Test for checking duplicate docket IDs in Qdrant collection."""

import os
import sys
import yaml
from collections import Counter

# Add parent directory to path to import from data-ingestion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data-ingestion'))

# Try to import dependencies, provide helpful error if missing
try:
    from qdrant_client import QdrantClient
except ImportError as e:
    print("\nError: Required dependencies not installed.")
    print("Please install dependencies from requirements-base.txt:")
    print("  pip install -r requirements-base.txt")
    print("\nOr install required packages directly:")
    print("  pip install qdrant-client pyyaml")
    sys.exit(1)


def get_qdrant_client():
    """Get a Qdrant client instance."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    return client


def load_config():
    """Load configuration from config.yml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_all_chunk_ids_from_qdrant(qdrant_client, collection_name):
    """
    Retrieve all chunk IDs (payload 'id' field) from the Qdrant collection.

    Args:
        qdrant_client: QdrantClient instance
        collection_name: Name of the collection to query

    Returns:
        list: List of all chunk IDs (may contain duplicates)
    """
    chunk_ids = []
    next_page = None

    print(f"Connecting to Qdrant collection: {collection_name}")

    try:
        collection = qdrant_client.get_collection(collection_name)
        print(f"Collection found with {collection.points_count} points")

        if collection.points_count == 0:
            print("Collection is empty - no chunk IDs to check")
            return chunk_ids

    except Exception as e:
        print(f"Error accessing collection '{collection_name}': {e}")
        return chunk_ids

    # Scroll through all points in the collection
    while True:
        try:
            points, next_page = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                limit=1000,
                offset=next_page
            )

            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("id")  # This is the unique chunk identifier
                if chunk_id:
                    chunk_ids.append(chunk_id)

            print(f"Retrieved {len(chunk_ids)} chunk IDs so far...")

            if not next_page:
                break

        except Exception as e:
            print(f"Error while scrolling collection: {e}")
            break

    print(f"Total chunk IDs retrieved: {len(chunk_ids)}")
    return chunk_ids


def check_for_duplicates(chunk_ids):
    """
    Check if there are duplicate chunk IDs in the list.

    Args:
        chunk_ids: List of chunk IDs

    Returns:
        tuple: (has_duplicates: bool, duplicate_info: dict)
    """
    counter = Counter(chunk_ids)
    duplicates = {chunk_id: count for chunk_id, count in counter.items() if count > 1}

    has_duplicates = len(duplicates) > 0

    duplicate_info = {
        'total_chunk_ids': len(chunk_ids),
        'unique_chunk_ids': len(counter),
        'duplicate_count': len(duplicates),
        'duplicates': duplicates
    }

    return has_duplicates, duplicate_info

def test_no_duplicate_chunk_ids():
    """
    Test that verifies there are no duplicate chunk IDs in the Qdrant collection.

    This test:
    1. Connects to the Qdrant client (Docker instance)
    2. Retrieves all chunk IDs (payload 'id' field) from the collection specified in config.yml
    3. Checks if there are any duplicates
    4. Reports detailed information about any duplicates found

    Note: The 'id' field in the payload should be unique for each chunk.
    The 'docket_id' field is expected to have duplicates (multiple chunks per docket).
    """
    # Load configuration
    config = load_config()
    collection_name = config['qdrant']['collection_name']

    print(f"\n{'='*60}")
    print(f"Testing for Duplicate Chunk IDs")
    print(f"{'='*60}")
    print(f"Collection: {collection_name}")

    # Connect to Qdrant
    try:
        qdrant_client = get_qdrant_client()
        print(f"✓ Connected to Qdrant at {os.getenv('QDRANT_URL', 'http://localhost:6333')}")
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        assert False, f"Could not connect to Qdrant: {e}"

    # Retrieve all chunk IDs
    chunk_ids = get_all_chunk_ids_from_qdrant(qdrant_client, collection_name)

    if not chunk_ids:
        print("\n⚠ No chunk IDs found in collection - nothing to test")
        return

    # Check for duplicates
    has_duplicates, duplicate_info = check_for_duplicates(chunk_ids)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Total chunk IDs: {duplicate_info['total_chunk_ids']}")
    print(f"Unique chunk IDs: {duplicate_info['unique_chunk_ids']}")
    print(f"Duplicate chunk IDs: {duplicate_info['duplicate_count']}")

    if has_duplicates:
        print(f"\n✗ DUPLICATES FOUND:")
        print(f"{'='*60}")
        for chunk_id, count in sorted(duplicate_info['duplicates'].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  Chunk ID '{chunk_id}': appears {count} times")
        print(f"{'='*60}\n")
    else:
        print(f"\n✓ No duplicates found - all chunk IDs are unique!\n")

    # Assert no duplicates
    assert not has_duplicates, \
        f"Found {duplicate_info['duplicate_count']} duplicate chunk IDs in collection '{collection_name}'"


if __name__ == "__main__":
    test_no_duplicate_chunk_ids()
