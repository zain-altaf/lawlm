"""
Processing Pipeline for Legal Case Search System.

This script orchestrates the processing of raw legal case data
for different search methods using specialized processors.
"""

import json
import os
import argparse
from typing import List, Dict, Any

from processing import KeywordProcessor, VectorProcessor, HybridProcessor


def load_raw_data(filename: str) -> List[Dict[str, Any]]:
    """Load raw case data from JSON file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} raw documents from {filename}")
        return docs
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run data ingestion first.")
        print("Run: python data_ingestion.py")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filename}")
        exit(1)


def process_keyword_search(docs: List[Dict[str, Any]], output_dir: str) -> str:
    """Process documents for keyword search."""
    processor = KeywordProcessor(output_dir=output_dir)
    return processor.process(docs)


def process_vector_search(docs: List[Dict[str, Any]]) -> tuple:
    """Process documents for vector search."""
    processor = VectorProcessor()
    collection_name = processor.process(docs)
    return collection_name, processor.get_embedder()


def process_hybrid_search(docs: List[Dict[str, Any]], shared_embedder=None) -> str:
    """Process documents for hybrid search."""
    processor = HybridProcessor(embedder=shared_embedder)
    return processor.process(docs)


def save_processing_summary(output_dir: str, results: Dict[str, Any]):
    """Save a summary of processing results."""
    summary_file = os.path.join(output_dir, "processing_summary.json")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing summary saved to: {summary_file}")


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description='Process legal case data for search methods')
    parser.add_argument('--input', default='data/raw_cases.json', 
                       help='Input raw data file (default: data/raw_cases.json)')
    parser.add_argument('--output_dir', default='data/processed', 
                       help='Output directory (default: data/processed)')
    parser.add_argument('--skip_keyword', action='store_true', 
                       help='Skip keyword search processing')
    parser.add_argument('--skip_vector', action='store_true', 
                       help='Skip vector search processing')
    parser.add_argument('--skip_hybrid', action='store_true', 
                       help='Skip hybrid search processing')
    
    args = parser.parse_args()

    print("Legal Case Data Processing Pipeline")
    print("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw data
    docs = load_raw_data(args.input)
    
    # Track processing results
    results = {
        "input_file": args.input,
        "total_documents": len(docs),
        "processed_methods": [],
        "outputs": {}
    }

    # Shared embedder for efficiency
    shared_embedder = None

    # Process keyword search (TF-IDF)
    if not args.skip_keyword:
        print("\n" + "="*20 + " KEYWORD SEARCH " + "="*20)
        try:
            keyword_file = process_keyword_search(docs, args.output_dir)
            results["processed_methods"].append("keyword")
            results["outputs"]["keyword"] = {
                "type": "json_file",
                "path": keyword_file,
                "description": "TF-IDF keyword search via minsearch"
            }
        except Exception as e:
            print(f"Error processing keyword search: {e}")
    else:
        print("\nSkipping keyword search processing")

    # Process vector search (Dense embeddings)
    if not args.skip_vector:
        print("\n" + "="*20 + " VECTOR SEARCH " + "="*21)
        try:
            vector_collection, embedder = process_vector_search(docs)
            shared_embedder = embedder  # Save for hybrid search
            results["processed_methods"].append("vector")
            results["outputs"]["vector"] = {
                "type": "qdrant_collection",
                "collection_name": vector_collection,
                "description": "Dense vector embeddings for semantic search"
            }
        except Exception as e:
            print(f"Error processing vector search: {e}")
    else:
        print("\nSkipping vector search processing")

    # Process hybrid search (Dense + Sparse)
    if not args.skip_hybrid:
        print("\n" + "="*20 + " HYBRID SEARCH " + "="*21)
        try:
            hybrid_collection = process_hybrid_search(docs, shared_embedder)
            results["processed_methods"].append("hybrid")
            results["outputs"]["hybrid"] = {
                "type": "qdrant_collection",
                "collection_name": hybrid_collection,
                "description": "Dense + BM25 sparse vectors with RRF"
            }
        except Exception as e:
            print(f"Error processing hybrid search: {e}")
    else:
        print("\nSkipping hybrid search processing")

    # Save processing summary
    save_processing_summary(args.output_dir, results)

    # Final summary
    print("\n" + "=" * 50)
    print("Processing Pipeline Completed Successfully!")
    print(f"\nProcessed {len(docs)} documents for {len(results['processed_methods'])} search method(s)")
    
    for method in results["processed_methods"]:
        output_info = results["outputs"][method]
        if output_info["type"] == "json_file":
            print(f"✓ {method.title()} Search: {output_info['path']}")
        elif output_info["type"] == "qdrant_collection":
            print(f"✓ {method.title()} Search: Qdrant collection '{output_info['collection_name']}'")
    
    print(f"\nNext step: Run 'python main.py' to start searching")
    print("Configure SEARCH_METHOD in your .env file:")
    if "keyword" in results["processed_methods"]:
        print("  SEARCH_METHOD=minsearch")
    if "vector" in results["processed_methods"]:
        print("  SEARCH_METHOD=vectorsearch")
    if "hybrid" in results["processed_methods"]:
        print("  SEARCH_METHOD=hybridsearch")


if __name__ == "__main__":
    main()