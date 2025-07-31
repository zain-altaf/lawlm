"""
Keyword Processor for TF-IDF based search using minsearch.

This processor handles data preparation for keyword-based search,
which uses TF-IDF internally within minsearch for ranking.
"""

import json
import os
from typing import List, Dict, Any


class KeywordProcessor:
    """Processes documents for keyword search with TF-IDF ranking."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "keyword_search.json")
        
    def process(self, documents: List[Dict[str, Any]]) -> str:
        """
        Process documents for keyword search.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            str: Path to the processed data file
        """
        print("\n--- Processing for Keyword Search (TF-IDF) ---")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For keyword search, we just need clean JSON data
        # minsearch will handle TF-IDF indexing internally
        processed_docs = []
        
        for doc in documents:
            # Ensure we have the required text content
            if not doc.get("opinion_text"):
                continue
                
            # Keep all document metadata for search and display
            processed_doc = {
                "id": doc.get("id"),
                "docket_number": doc.get("docket_number"),
                "case_name": doc.get("case_name"),
                "court_id": doc.get("court_id"),
                "judges": doc.get("judges"),
                "author": doc.get("author"),
                "type": doc.get("type"),
                "sha1": doc.get("sha1"),
                "download_url": doc.get("download_url"),
                "opinion_text": doc.get("opinion_text"),
                "source_field": doc.get("source_field")
            }
            
            processed_docs.append(processed_doc)
        
        # Save processed documents
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(processed_docs, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {len(processed_docs)} documents for keyword search")
        print(f"Saved to: {self.output_file}")
        print("Note: TF-IDF indexing will be performed by minsearch at search time")
        
        return self.output_file
    
    def get_search_config(self) -> Dict[str, Any]:
        """
        Get configuration for keyword search.
        
        Returns:
            dict: Configuration for minsearch
        """
        return {
            "text_fields": ["opinion_text"],
            "keyword_fields": ["docket_number", "court_id", "judges", "type"],
            "boost_dict": {"opinion_text": 1.5},
            "filter_dict": {"court_id": "scotus"}
        }