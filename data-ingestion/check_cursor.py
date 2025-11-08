"""
Script to check for non-null cursor values in Qdrant collection.
"""

from dotenv import load_dotenv
import os
from opinion_utills import get_qdrant_client

load_dotenv()

def check_cursors():
    """
    Scan all points in the Qdrant collection and find any with non-empty cursor values.
    """
    collection_name = "caselaw-chunks-scotus"

    # Get Qdrant client
    qdrant_client = get_qdrant_client()

    print(f"Scanning collection: {collection_name}")
    print("=" * 60)

    # Track statistics
    total_points = 0
    points_with_cursor = []
    next_page = None

    # Scroll through all points
    while True:
        points, next_page = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=1000,
            offset=next_page
        )

        for point in points:
            total_points += 1
            payload = point.payload or {}
            cursor = payload.get('cursor', '')

            # Check if cursor is non-empty
            if cursor:
                points_with_cursor.append({
                    'point_id': point.id,
                    'docket_id': payload.get('docket_id'),
                    'chunk_id': payload.get('chunk_id'),
                    'time_processed': payload.get('time_processed'),
                    'cursor': cursor
                })

        if not next_page:
            break

    # Print results
    print(f"Total points scanned: {total_points}")
    print(f"Points with non-empty cursor: {len(points_with_cursor)}")
    print("=" * 60)

    if points_with_cursor:
        print("\nPoints with cursor values:")
        for p in points_with_cursor:
            print(f"\nPoint ID: {p['point_id']}")
            print(f"  Docket ID: {p['docket_id']}")
            print(f"  Chunk ID: {p['chunk_id']}")
            print(f"  Time Processed: {p['time_processed']}")
            print(f"  Cursor: {p['cursor']}")
    else:
        print("\nNo points found with cursor values.")

    return points_with_cursor


if __name__ == "__main__":
    check_cursors()
