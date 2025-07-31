import minsearch
import requests
import os
import uuid
import json
import re

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
CASELAW_API_KEY = os.getenv("CASELAW_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEARCH_METHOD = os.getenv("SEARCH_METHOD", "minsearch") # Default to minsearch
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# --- Config for Standard Vector Search ---
# We will use this model for BOTH vector and hybrid search now
DENSE_MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_SIZE = 384 # Dimension of 'all-MiniLM-L6-v2'

EMBEDDER = SentenceTransformer(DENSE_MODEL_NAME)
VECTOR_COLLECTION_NAME = "caselaw-cases-dense"

# --- Config for Hybrid Search (Dense + Sparse with RRF) ---
HYBRID_COLLECTION_NAME = "caselaw-hybrid-rrf-minilm"
# The sparse model remains the same
HYBRID_SPARSE_MODEL = "Qdrant/bm25"

# General settings
NUM_RESULTS = 5
CHUNK_SIZE = 500 # Max number of words per chunk

# --- API Clients and Headers ---
HEADERS = {"Authorization": f"Token {CASELAW_API_KEY}"}
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# --- Data Fetching and Processing ---
# (This section is unchanged)

def clean_text(content: str) -> str:
    """Strips HTML/XML tags and normalizes whitespace."""
    if not content:
        return ''
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text)

def chunk_text(text: str, max_words: int = CHUNK_SIZE) -> list[str]:
    """Splits text into chunks of a maximum number of words."""
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def process_docket(court: str = 'scotus', num_dockets: int = 5) -> list:
    """
    Fetches dockets, clusters, and opinions from CourtListener API,
    returning a simplified list of dictionaries.
    """
    print(f"Fetching {num_dockets} dockets from court: {court}...")
    docket_resp = requests.get(
        "https://www.courtlistener.com/api/rest/v4/dockets/",
        params={"court": court, "page_size": num_dockets},
        headers=HEADERS
    )
    docket_resp.raise_for_status()
    dockets = docket_resp.json().get("results", [])

    all_data = []
    for docket in dockets:
        for cluster_url in docket.get("clusters", []):
            cluster_resp = requests.get(cluster_url, headers=HEADERS)
            cluster = cluster_resp.json()
            for opinion_url in cluster.get("sub_opinions", []):
                opinion_resp = requests.get(opinion_url, headers=HEADERS)
                opinion = opinion_resp.json()
                text, source = "", 'Unknown'
                for field in ['html_with_citations', 'html_columbia', 'html_lawbox', 'xml_harvard', 'html', 'plain_text']:
                    if opinion.get(field):
                        text = clean_text(opinion[field]) if 'html' in field or 'xml' in field else re.sub(r'\s+', ' ', opinion[field].strip())
                        source = field
                        break
                all_data.append({
                    "id": opinion.get("id"),
                    "docket_number": docket.get("docket_number"),
                    "case_name": cluster.get("case_name"),
                    "court_id": docket.get("court_id"),
                    "judges": cluster.get("judges"),
                    "author": opinion.get("author_str"),
                    "type": opinion.get("type"),
                    "sha1": opinion.get("sha1"),
                    "download_url": opinion.get("download_url"),
                    "opinion_text": text
                })
    print(f"Successfully processed {len(all_data)} opinions.")
    return all_data

# --- Search and Indexing ---
# (get_qdrant_client, run_minsearch, and run_vectorsearch are unchanged)

def get_qdrant_client() -> QdrantClient:
    """Initializes and returns a Qdrant client."""
    try:
        client = QdrantClient(QDRANT_URL)
        client.get_collections()
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant at {QDRANT_URL}. Is the server running?")
        print(f"Details: {e}")
        exit()

def run_minsearch(docs: list, query: str) -> list:
    """Performs a keyword search using minsearch and returns results."""
    print("\n--- Running Minsearch ---")
    index = minsearch.Index(
        text_fields=['opinion_text'],
        keyword_fields=['docket_number', 'court_id', 'judges', 'type']
    )
    index.fit(docs)
    results = index.search(
        query=query,
        filter_dict={"court_id": "scotus"},
        boost_dict={"opinion_text": 1.5},
        num_results=NUM_RESULTS
    )
    return results

def run_vectorsearch(client: QdrantClient, docs: list, query: str) -> list:
    """Performs vector search using Qdrant and returns results."""
    print("\n--- Running Vector Search ---")
    try:
        client.create_collection(
            collection_name=VECTOR_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )
        print(f"Collection '{VECTOR_COLLECTION_NAME}' created.")
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=EMBEDDER.encode(doc['opinion_text']).tolist(),
                payload=doc
            ) for doc in docs if doc.get('opinion_text')
        ]
        client.upsert(collection_name=VECTOR_COLLECTION_NAME, points=points, wait=True)
        print(f"Upserted {len(points)} documents.")
    except Exception:
        print(f"Collection '{VECTOR_COLLECTION_NAME}' already exists. Skipping creation/upsert.")

    query_vector = EMBEDDER.encode(query).tolist()
    hits = client.search(
        collection_name=VECTOR_COLLECTION_NAME,
        query_vector=query_vector,
        limit=NUM_RESULTS,
        with_payload=True
    )
    return [hit.payload for hit in hits]

def run_hybrid_search(client: QdrantClient, docs: list, query: str) -> list:
    """
    Performs hybrid search using Qdrant with on-the-fly embedding and RRF.
    This version now uses all-MiniLM-L6-v2 for dense vectors.
    """
    print("\n--- Running Hybrid Search with RRF (using MiniLM model) ---")

    # 1. Recreate collection with dense (MiniLM) and sparse vector configs
    try:
        # Use client.collection_exists and client.create_collection instead of recreate_collection
        if client.collection_exists(collection_name=HYBRID_COLLECTION_NAME):
            client.delete_collection(collection_name=HYBRID_COLLECTION_NAME)
            print(f"Existing collection '{HYBRID_COLLECTION_NAME}' deleted.")

        client.create_collection(
            collection_name=HYBRID_COLLECTION_NAME,
            vectors_config={
                # Vector size is now 384 for MiniLM
                "dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        )
        print(f"Collection '{HYBRID_COLLECTION_NAME}' created for hybrid search.")

        # 2. Prepare and upsert points with on-the-fly embedding
        points = []
        for doc in docs:
            opinion_text = doc.get("opinion_text")
            if not opinion_text:
                continue
            
            for chunk in chunk_text(opinion_text):
                chunk_payload = {**doc, "opinion_text": chunk}
                
                # Manually embed the dense vector using the pre-loaded EMBEDDER
                dense_vector = EMBEDDER.encode(chunk).tolist()

                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vector, # Use the pre-computed dense vector
                            "sparse": models.Document(text=chunk, model=HYBRID_SPARSE_MODEL), # Sparse model can still use on-the-fly
                        },
                        payload=chunk_payload
                    )
                )

        client.upsert(collection_name=HYBRID_COLLECTION_NAME, points=points, wait=True)
        print(f"Upserted {len(points)} document chunks for hybrid search.")

    except Exception as e:
        print(f"An error occurred with the collection '{HYBRID_COLLECTION_NAME}': {e}")
        print("Skipping creation/upsert. The search might fail or use old data.")

    # 3. Perform search using Reciprocal Rank Fusion (RRF)
    print("Performing RRF search...")

    # Manually embed the dense query vector using the pre-loaded EMBEDDER
    dense_query_vector = EMBEDDER.encode(query).tolist()

    results = client.query_points(
        collection_name=HYBRID_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                # Use the pre-computed dense query vector
                query=dense_query_vector,
                using="dense",
                limit=(5 * NUM_RESULTS),
            ),
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

# --- RAG and LLM Interaction ---
# (perform_rag_with_llm is unchanged)

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
        print(f"[{idx}] Case: {res.get('case_name')} | Author: {res.get('author')} | URL: {res.get('download_url')}")

    try:
        selected_indices_input = input("\nEnter indices of cases to include in context (comma-separated), or press Enter to cancel: ")
        if not selected_indices_input:
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
                f"Case Name: {res.get('case_name')}\n"
                f"Opinion Type: {res.get('type')}\n"
                f"Case Text: {res.get('opinion_text')}\n\n"
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
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    print("\nModel Response:\n")
    print(response.choices[0].message.content.strip())


# --- Main Execution ---
# (main is unchanged)

def main():
    """Main function to run the legal RAG application."""
    docs = process_docket(court='scotus', num_dockets=5)
    with open("cases_processed.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    query = input("Please enter your search query: ")

    if SEARCH_METHOD == "minsearch":
        results = run_minsearch(docs, query)
        perform_rag_with_llm(results, query)
    elif SEARCH_METHOD in ["vectorsearch", "hybridsearch"]:
        client = get_qdrant_client()
        if SEARCH_METHOD == "vectorsearch":
            results = run_vectorsearch(client, docs, query)
        else: # hybridsearch
            results = run_hybrid_search(client, docs, query)
        perform_rag_with_llm(results, query)
    else:
        print(f"Invalid SEARCH_METHOD: '{SEARCH_METHOD}'. Please use 'minsearch', 'vectorsearch', or 'hybridsearch'.")

if __name__ == "__main__":
    main()