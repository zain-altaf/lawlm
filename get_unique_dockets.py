#!/usr/bin/env python3
"""
Get all unique docket numbers from Qdrant collection and check for duplicates.
"""
import os
from typing import Set, Dict, List
from collections import defaultdict
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def analyze_docket_numbers(collection_name: str = "caselaw-chunks-hybrid") -> Dict[str, any]:
    """Analyze docket numbers in the collection and detect duplicates."""
    
    # Connect to local Qdrant
    client = QdrantClient(url="http://localhost:6333")
    
    # Check if collection exists
    if not client.collection_exists(collection_name=collection_name):
        print(f"❌ Collection '{collection_name}' does not exist")
        return {}
    
    docket_counts = defaultdict(int)
    docket_points = defaultdict(list)  # Track point IDs for each docket
    chunk_combinations = defaultdict(int)  # Track (docket_number, chunk_index) combinations
    duplicate_chunks = defaultdict(list)  # Track actual duplicate chunks
    offset = None
    total_points = 0
    
    print(f"🔍 Scanning collection '{collection_name}' for docket number analysis...")
    
    while True:
        # Scroll through all points in the collection
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False  # We only need payload data
        )
        
        if not points:
            break
            
        # Count docket numbers and track point IDs
        for point in points:
            total_points += 1
            docket_number = point.payload.get('docket_number', '')
            chunk_index = point.payload.get('chunk_index', '')
            
            if docket_number:
                docket_counts[docket_number] += 1
                point_data = {
                    'point_id': point.id,
                    'document_id': point.payload.get('document_id', ''),
                    'case_name': point.payload.get('case_name', ''),
                    'chunk_index': chunk_index
                }
                docket_points[docket_number].append(point_data)
                
                # Track chunk combinations for true duplicate detection
                chunk_key = (docket_number, chunk_index)
                chunk_combinations[chunk_key] += 1
                duplicate_chunks[chunk_key].append(point_data)
                
        offset = next_offset
        if next_offset is None:
            break
            
        # Progress update
        if total_points % 10000 == 0:
            print(f"📊 Processed {total_points} points, found {len(docket_counts)} unique dockets...")
    
    # Analyze real duplicates (same docket_number + chunk_index)
    real_duplicates = {key: count for key, count in chunk_combinations.items() if count > 1}
    duplicates = {f"{key[0]} (chunk {key[1]})": count for key, count in real_duplicates.items()}
    unique_dockets = set(docket_counts.keys())
    
    print(f"\n📊 Docket Number Analysis:")
    print(f"   Total points in collection: {total_points}")
    print(f"   Unique docket numbers: {len(unique_dockets)}")
    print(f"   Duplicate chunk combinations: {len(duplicates)}")
    
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} duplicate chunk combinations (same docket_number + chunk_index):")
        total_duplicate_points = sum(duplicates.values())
        print(f"   Total points from duplicated chunks: {total_duplicate_points}")
        
        # Show first duplicate found
        first_duplicate_key = next(iter(real_duplicates.keys()))
        first_duplicate_count = real_duplicates[first_duplicate_key]
        print(f"\n🔍 First duplicate found:")
        print(f"   Docket: {first_duplicate_key[0]}")
        print(f"   Chunk Index: {first_duplicate_key[1]}")
        print(f"   Appears {first_duplicate_count} times")
        
        # Show the point IDs for this first duplicate
        first_duplicate_points = duplicate_chunks[first_duplicate_key]
        print(f"   Point IDs:")
        for i, point in enumerate(first_duplicate_points):
            print(f"     [{i+1}] Point ID: {point['point_id']}, Doc ID: {point['document_id']}")
        
        # Show top duplicates
        top_duplicates = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🔍 Top 10 chunk combinations with most duplicates:")
        for chunk_combo, count in top_duplicates:
            print(f"   {chunk_combo}: {count} duplicates")
    else:
        print(f"\n✅ No duplicate chunk combinations found - each (docket_number, chunk_index) pair is unique")
    
    return {
        'total_points': total_points,
        'unique_dockets': unique_dockets,
        'duplicate_chunks': duplicates,
        'real_duplicates': dict(real_duplicates),
        'duplicate_chunks_raw': dict(duplicate_chunks),
        'docket_points': dict(docket_points),
        'docket_counts': dict(docket_counts)
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze docket numbers in Qdrant collection")
    parser.add_argument('--collection', default='caselaw-chunks-hybrid', help='Collection name')
    parser.add_argument('--list', action='store_true', help='List all unique docket numbers')
    parser.add_argument('--duplicates', action='store_true', help='Show detailed duplicate analysis')
    parser.add_argument('--docket', type=str, help='Show details for a specific docket number')
    
    args = parser.parse_args()
    
    analysis = analyze_docket_numbers(args.collection)
    
    if not analysis:
        return
    
    if args.list:
        print(f"\n📋 All unique docket numbers:")
        for docket in sorted(analysis['unique_dockets']):
            print(f"   {docket}")
    
    if args.duplicates and analysis['duplicate_chunks']:
        print(f"\n🔍 Detailed duplicate chunk analysis:")
        for chunk_combo, count in sorted(analysis['duplicate_chunks'].items(), key=lambda x: x[1], reverse=True):
            print(f"\n   Chunk combination: {chunk_combo} ({count} duplicates)")
            # Extract docket_number and chunk_index from the real_duplicates key
            for key, raw_count in analysis['real_duplicates'].items():
                formatted_key = f"{key[0]} (chunk {key[1]})"
                if formatted_key == chunk_combo:
                    points = analysis['duplicate_chunks_raw'][key]
                    for i, point in enumerate(points[:5]):  # Show first 5 duplicates
                        print(f"     [{i+1}] Point ID: {point['point_id']}, Doc ID: {point['document_id']}")
                    if len(points) > 5:
                        print(f"     ... and {len(points) - 5} more duplicates")
                    break
    
    if args.docket:
        docket = args.docket
        if docket in analysis['docket_points']:
            points = analysis['docket_points'][docket]
            print(f"\n🔍 Details for docket: {docket}")
            print(f"   Total chunks: {len(points)}")
            if points:
                print(f"   Case name: {points[0]['case_name']}")
                print(f"   Chunks:")
                for point in points:
                    print(f"     Point ID: {point['point_id']}, Doc ID: {point['document_id']}, Chunk: {point['chunk_index']}")
        else:
            print(f"❌ Docket '{docket}' not found in collection")

if __name__ == "__main__":
    main()