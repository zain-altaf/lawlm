import minsearch
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEARCH_METHOD = os.getenv("SEARCH_METHOD", "minsearch")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# --- Search Method Configurations ---
# Shared vector configuration
DENSE_MODEL_NAME = 'all-MiniLM-L6-v2'

# Collection names (must match processors)
VECTOR_COLLECTION_NAME = "caselaw-vector-search"
HYBRID_COLLECTION_NAME = "caselaw-hybrid-search"
HYBRID_SPARSE_MODEL = "Qdrant/bm25"

# Settings
NUM_RESULTS = 5

# --- Initialize Components ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Only load embedder for vector/hybrid search
EMBEDDER = None
if SEARCH_METHOD in ["vectorsearch", "hybridsearch"]:
    print("Loading embedding model for search...")
    EMBEDDER = SentenceTransformer(DENSE_MODEL_NAME)

def load_processing_summary(summary_file: str = "data/processed/processing_summary.json") -> dict:
    """Load processing summary to understand what's available."""
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return summary
    except FileNotFoundError:
        print(f"Warning: {summary_file} not found.")
        print("Please run the processing pipeline first: python process_data.py")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {summary_file}")
        return {}

def load_processed_data_for_keyword(summary: dict) -> list:
    """Load processed case data for keyword search."""
    if "keyword" not in summary.get("outputs", {}):
        print("Error: Keyword search data not found.")
        print("Please run: python process_data.py")
        exit(1)
    
    keyword_output = summary["outputs"]["keyword"]
    if keyword_output["type"] != "json_file":
        print("Error: Invalid keyword output type.")
        exit(1)
    
    filename = keyword_output["path"]
    try:
        with open(filename, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} processed documents from {filename}")
        return docs
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run data processing first.")
        print("Run: python process_data.py")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filename}")
        exit(1)

def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client for vector/hybrid search."""
    try:
        client = QdrantClient(QDRANT_URL)
        client.get_collections()
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant at {QDRANT_URL}. Is the server running?")
        print(f"Details: {e}")
        exit(1)

def run_keyword_search(docs: list, query: str) -> list:
    """
    Performs keyword search using minsearch with TF-IDF.
    """
    print("\n--- Running Keyword Search (TF-IDF) ---")
    
    # Create minsearch index with TF-IDF
    index = minsearch.Index(
        text_fields=['opinion_text'],
        keyword_fields=['docket_number', 'court_id', 'judges', 'type']
    )
    
    # Fit the index (builds TF-IDF internally)
    index.fit(docs)
    
    # Search with TF-IDF scoring
    results = index.search(
        query=query,
        filter_dict={"court_id": "scotus"},
        boost_dict={"opinion_text": 1.5},
        num_results=NUM_RESULTS
    )
    
    return results

def run_vector_search(client: QdrantClient, query: str) -> list:
    """
    Performs pure vector search using pre-processed embeddings.
    """
    print("\n--- Running Vector Search ---")
    
    # Check if collection exists
    if not client.collection_exists(collection_name=VECTOR_COLLECTION_NAME):
        print(f"Error: Collection '{VECTOR_COLLECTION_NAME}' does not exist.")
        print("Please run data processing first: python process_data.py")
        return []

    # Create query embedding using shared embedder
    query_vector = EMBEDDER.encode(query).tolist()
    
    # Search in pre-processed vector collection
    hits = client.search(
        collection_name=VECTOR_COLLECTION_NAME,
        query_vector=query_vector,
        limit=NUM_RESULTS,
        with_payload=True
    )
    
    return [hit.payload for hit in hits]

def run_hybrid_search(client: QdrantClient, query: str) -> list:
    """
    Performs hybrid search using pre-processed dense vectors + BM25 sparse vectors.
    """
    print("\n--- Running Hybrid Search (Vector + BM25) ---")
    
    # Check if collection exists
    if not client.collection_exists(collection_name=HYBRID_COLLECTION_NAME):
        print(f"Error: Collection '{HYBRID_COLLECTION_NAME}' does not exist.")
        print("Please run data processing first: python process_data.py")
        return []

    # Create dense query embedding using shared embedder
    dense_query_vector = EMBEDDER.encode(query).tolist()

    # Perform hybrid search with RRF (Reciprocal Rank Fusion)
    results = client.query_points(
        collection_name=HYBRID_COLLECTION_NAME,
        prefetch=[
            # Dense vector search (using shared embeddings)
            models.Prefetch(
                query=dense_query_vector,
                using="dense",
                limit=(5 * NUM_RESULTS),
            ),
            # Sparse BM25 search
            models.Prefetch(
                query=models.Document(text=query, model=HYBRID_SPARSE_MODEL),
                using="sparse",
                limit=(5 * NUM_RESULTS),
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=NUM_RESULTS,
        with_payload=True,
    )

    return [hit.payload for hit in results.points]

def perform_rag_with_llm(results: list, original_query: str):
    """
    Handles displaying results, user selection, context building,
    and generation with the LLM.
    """
    if not results:
        print("No relevant documents found. Please try a different query.")
        return

    print("\nTop Retrieved Cases:\n")
    for idx, res in enumerate(results):
        print(f"[{idx}] Case: {res.get('case_name', 'Unknown')} | Author: {res.get('author', 'Unknown')} | URL: {res.get('download_url', 'N/A')}")

    try:
        selected_indices_input = input("\nEnter indices of cases to include in context (comma-separated), or press Enter to cancel: ")
        if not selected_indices_input.strip():
            print("No cases selected. Exiting.")
            return
        selected_indices = [int(i.strip()) for i in selected_indices_input.split(",") if i.strip().isdigit()]
    except (ValueError, IndexError):
        print("Invalid input. Please enter valid indices.")
        return

    context_parts = []
    for idx in selected_indices:
        if 0 <= idx < len(results):
            res = results[idx]
            context_parts.append(
                f"Case Name: {res.get('case_name', 'Unknown')}\n"
                f"Opinion Type: {res.get('type', 'Unknown')}\n"
                f"Case Text: {res.get('opinion_text', 'No text available')}\n\n"
            )
    context = "".join(context_parts)

    if not context:
        print("No valid cases were selected to build context. Exiting.")
        return

    generation_question = input(f"\nEnter the QUESTION for the LLM (press Enter to use the original query): ") or original_query
    prompt = (
        "You are a legal research assistant. Answer the QUESTION using only the CONTEXT provided below.\n"
        "If the CONTEXT lacks the necessary information, state that you cannot answer the question based on the provided text.\n\n"
        f"QUESTION: {generation_question}\n\n"
        "CONTEXT:\n"
        f"{context}"
    )

    print("\n--- Generating Response ---")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        print("\nModel Response:\n")
        print(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error generating response: {e}")

def main():
    """Main function to run the legal search application."""
    print("Legal Search Application")
    print("=" * 30)
    print(f"Search method: {SEARCH_METHOD}")
    
    # Load processing summary to understand what's available
    summary = load_processing_summary()
    
    # Check if the requested search method was processed
    processed_methods = summary.get("processed_methods", [])
    
    # Map search methods to processing method names
    method_mapping = {
        "minsearch": "keyword",
        "vectorsearch": "vector", 
        "hybridsearch": "hybrid"
    }
    
    required_method = method_mapping.get(SEARCH_METHOD)
    if required_method and required_method not in processed_methods:
        print(f"\nError: {SEARCH_METHOD} has not been processed yet.")
        print(f"Available methods: {processed_methods}")
        print("\nPlease run the processing pipeline first:")
        print("python process_data.py")
        return
    
    # Get user query
    query = input("\nPlease enter your search query: ")
    if not query.strip():
        print("Empty query provided. Exiting.")
        return

    # Route to appropriate search method
    results = []
    
    if SEARCH_METHOD == "minsearch":
        # Keyword search with TF-IDF
        docs = load_processed_data_for_keyword(summary)
        results = run_keyword_search(docs, query)
        
    elif SEARCH_METHOD == "vectorsearch":
        # Pure vector search using pre-processed embeddings
        if EMBEDDER is None:
            print("Error: Embedder not initialized for vector search.")
            return
        client = get_qdrant_client()
        results = run_vector_search(client, query)
        
    elif SEARCH_METHOD == "hybridsearch":
        # Hybrid search using pre-processed vectors + BM25
        if EMBEDDER is None:
            print("Error: Embedder not initialized for hybrid search.")
            return
        client = get_qdrant_client()
        results = run_hybrid_search(client, query)
        
    else:
        print(f"Invalid SEARCH_METHOD: '{SEARCH_METHOD}'")
        print("Valid options: 'minsearch', 'vectorsearch', 'hybridsearch'")
        return

    # Continue with RAG pipeline
    perform_rag_with_llm(results, query)

if __name__ == "__main__":
    main()