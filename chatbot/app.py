#!/usr/bin/env python3
"""
Legal RAG Chatbot Service

A Flask-based REST API for legal document queries using hybrid search.
Combines semantic (dense) and keyword (BM25) search with OpenAI for summaries.
"""

import logging
import os
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import requests
import PyPDF2
import io

load_dotenv()

# Load config from project root (works both locally and in Docker)
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yml')
if not os.path.exists(config_path):
    # Fallback for Docker: config is mounted at /app/config.yml
    config_path = '/app/config.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration from environment and config file
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = config['qdrant']['collection_name']
EMBEDDING_MODEL = config['vectorization']['embedding_model']
OPENAI_MODEL = config['rag']['openai_model']
MAX_RESULTS = config['rag']['max_results']
PORT = config['services']['chatbot']['port']

# Initialize clients
logger.info(f"Initializing chatbot service...")
logger.info(f"Qdrant URL: {QDRANT_URL}")
logger.info(f"Collection: {COLLECTION_NAME}")
logger.info(f"Embedding model: {EMBEDDING_MODEL}")

qdrant_client = QdrantClient(url=QDRANT_URL)
logger.info("Connected to Qdrant")

# Load embedding model
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)
embedder.eval()
logger.info("Embedding model loaded")

# Initialize OpenAI client
OPENAI_AVAILABLE = False
openai_client = None
if os.getenv("OPENAI_API_KEY"):
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        OPENAI_AVAILABLE = True
        logger.info("OpenAI client initialized")
    except Exception as e:
        logger.warning(f"OpenAI setup failed: {e}")
else:
    logger.warning("OPENAI_API_KEY not set. Summaries will be unavailable.")


class LegalRAGService:
    """
    Legal RAG service for hybrid search and query processing.
    """

    def __init__(self):
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.collection_name = COLLECTION_NAME
        self.openai_model = OPENAI_MODEL
        self.max_results = MAX_RESULTS

        # BGE models work better with specific prompts
        if 'bge' in EMBEDDING_MODEL.lower():
            self.query_prefix = "Represent this query for searching relevant legal passages: "
        else:
            self.query_prefix = ""

        # Verify collection exists
        if not self.qdrant_client.collection_exists(self.collection_name):
            logger.error(f"Collection '{self.collection_name}' not found!")
            raise ValueError(f"Collection '{self.collection_name}' does not exist")

        # Get collection info
        info = self.qdrant_client.get_collection(self.collection_name)
        logger.info(f"Collection has {info.points_count} chunks")

    def hybrid_search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).

        Combines dense vector search (semantic) with sparse BM25 search (keyword).

        Args:
            query: Search query text
            limit: Number of results to return
            score_threshold: Minimum score threshold

        Returns:
            List of search results with scores and payloads
        """
        # Use defaults from config if not provided
        if limit is None:
            limit = config['rag']['max_results']
        if score_threshold is None:
            score_threshold = config['rag']['default_score_threshold']

        logger.info(f"Hybrid search: '{query}' (limit={limit}, threshold={score_threshold})")

        try:
            # Create enhanced query with BGE prefix if applicable
            enhanced_query = self.query_prefix + query if self.query_prefix else query

            # Create dense vector for the query
            query_vector = self.embedder.encode(enhanced_query).tolist()

            # Get RRF prefetch multiplier from config
            rrf_multiplier = config['rag']['rrf_prefetch_multiplier']

            # Perform hybrid search with RRF using prefetch
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Dense vector search (semantic)
                    models.Prefetch(
                        query=query_vector,
                        using="bge-small",
                        limit=(rrf_multiplier * limit),  # Fetch more for better fusion
                    ),
                    # Sparse vector search (keyword/BM25)
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model="Qdrant/bm25",
                        ),
                        using="bm25",
                        limit=(rrf_multiplier * limit),
                    ),
                ],
                # Use RRF to combine results
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            results = []
            for point in search_result.points:
                result = {
                    'id': point.id,
                    'score': point.score,
                    'payload': point.payload,
                    'search_type': 'hybrid_rrf'
                }
                results.append(result)

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into context for OpenAI.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant legal documents found."

        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result['payload']
            case_name = payload.get('case_name', 'Unknown Case')
            court = payload.get('court_id', 'Unknown Court')
            date_filed = payload.get('date_filed', 'Unknown Date')
            opinion_type = payload.get('opinion_type', '')
            text = payload.get('text', '')
            score = result.get('score', 0)

            # Preview first 500 characters
            text_preview = text.strip()[:500]

            context_part = f"""
Document {i} (Relevance: {score:.3f}):
Case: {case_name}
Court: {court.upper()}
Date: {date_filed}
Type: {opinion_type}
Content: {text_preview}{"..." if len(text) > 500 else ""}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def generate_summary(self, query: str, context: str) -> str:
        """
        Generate concise summary using OpenAI API.

        Args:
            query: Original user query
            context: Formatted search results

        Returns:
            Generated summary or fallback message
        """
        if not OPENAI_AVAILABLE or not openai_client:
            return f"OpenAI not available. Found {len(context.split('Document'))-1} relevant documents."

        system_prompt = """You are a legal research assistant. Based on the provided legal document excerpts, provide a concise and accurate summary that directly answers the user's question.

Guidelines:
- Keep response to exactly 150 words or less
- Focus on the most relevant legal information
- Include case names and key legal principles when relevant
- Be precise and authoritative
- If the documents don't fully answer the question, state what information is available
- Use clear, professional legal language"""

        user_prompt = f"""Query: {query}

Relevant Legal Documents:
{context}

Please provide a concise 150-word summary that answers the query based on these legal documents."""

        try:
            response = openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.1,
                top_p=0.95
            )

            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated {len(summary.split())} word summary")
            return summary

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating summary. Found relevant information about: {query}"

    def query(
        self,
        question: str,
        score_threshold: float = None,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG query: search + generate summary.

        Args:
            question: User's legal question
            score_threshold: Minimum relevance score (uses config default if None)
            max_results: Maximum number of results (uses config default if None)

        Returns:
            Dictionary with summary, sources, and metadata
        """
        start_time = datetime.now()

        # Use config defaults if not provided
        if max_results is None:
            max_results = config['rag']['max_results']
        if score_threshold is None:
            score_threshold = config['rag']['default_score_threshold']

        limit = max_results

        # Step 1: Search relevant documents
        try:
            search_results = self.hybrid_search(
                query=question,
                limit=limit,
                score_threshold=score_threshold
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "question": question,
                "summary": f"Search failed: {str(e)}",
                "sources": [],
                "search_type": "hybrid_rrf",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "documents_found": 0,
                "error": str(e)
            }

        if not search_results:
            return {
                "question": question,
                "summary": "No relevant legal documents found for this query. Try rephrasing your question or using different keywords.",
                "sources": [],
                "search_type": "hybrid_rrf",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "documents_found": 0
            }

        # Step 2: Format context for OpenAI
        context = self.format_search_results(search_results)

        # Step 3: Generate summary
        summary = self.generate_summary(question, context)

        # Step 4: Prepare sources information
        sources = []
        for result in search_results:
            payload = result['payload']
            source = {
                "case_name": payload.get('case_name', 'Unknown Case'),
                "court": payload.get('court_id', 'Unknown'),
                "date_filed": payload.get('date_filed', 'Unknown'),
                "opinion_type": payload.get('opinion_type', ''),
                "relevance_score": result.get('score', 0),
                "chunk_id": payload.get('chunk_id', ''),
                "text": payload.get('text', ''),
                "download_url": payload.get('download_url', '')
            }
            sources.append(source)

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "question": question,
            "summary": summary,
            "sources": sources,
            "search_type": "hybrid_rrf",
            "processing_time": processing_time,
            "documents_found": len(search_results)
        }


# Initialize RAG service
rag_service = LegalRAGService()


# API Routes

@app.route('/', methods=['GET'])
def index():
    """Serve the web interface."""
    return send_from_directory('static', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        # Check Qdrant connection
        collections = qdrant_client.get_collections()
        collection_exists = qdrant_client.collection_exists(COLLECTION_NAME)

        return jsonify({
            "status": "healthy",
            "qdrant_connected": True,
            "collection_exists": collection_exists,
            "collection_name": COLLECTION_NAME,
            "openai_available": OPENAI_AVAILABLE,
            "embedding_model": EMBEDDING_MODEL
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/query', methods=['POST'])
def query_endpoint():
    """
    Query endpoint for legal RAG - now just returns search results without summary.

    Request body:
    {
        "question": "What is the holding in Brown v. Board of Education?",
        "score_threshold": 0.4,  // optional, default 0.4
        "max_results": 3         // optional, default from env
    }

    Response:
    {
        "query": "...",
        "results": [...],
        "search_type": "hybrid_rrf",
        "processing_time": 0.45,
        "documents_found": 3
    }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data['question']
        score_threshold = data.get('score_threshold', None)
        max_results = data.get('max_results', None)

        logger.info(f"Received query: '{question}'")

        # Use hybrid_search directly instead of full RAG query
        start_time = datetime.now()
        results = rag_service.hybrid_search(
            query=question,
            limit=max_results,
            score_threshold=score_threshold
        )
        processing_time = (datetime.now() - start_time).total_seconds()

        return jsonify({
            "query": question,
            "results": results,
            "search_type": "hybrid_rrf",
            "processing_time": processing_time,
            "documents_found": len(results)
        }), 200

    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    Search endpoint for hybrid search only (no summarization).

    Request body:
    {
        "query": "Fourth Amendment search and seizure",
        "limit": 3,              // optional, default 3
        "score_threshold": 0.4   // optional, default 0.4
    }

    Response:
    {
        "query": "...",
        "results": [...],
        "search_type": "hybrid_rrf",
        "processing_time": 0.45,
        "documents_found": 3
    }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request body"
            }), 400

        query = data['query']
        limit = data.get('limit', None)
        score_threshold = data.get('score_threshold', None)

        logger.info(f"Received search: '{query}'")

        start_time = datetime.now()
        results = rag_service.hybrid_search(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )
        processing_time = (datetime.now() - start_time).total_seconds()

        return jsonify({
            "query": query,
            "results": results,
            "search_type": "hybrid_rrf",
            "processing_time": processing_time,
            "documents_found": len(results)
        }), 200

    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/collection/info', methods=['GET'])
def collection_info():
    """Get information about the Qdrant collection."""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)

        return jsonify({
            "collection_name": COLLECTION_NAME,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status
        }), 200

    except Exception as e:
        logger.error(f"Collection info error: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/case/fetch', methods=['POST'])
def fetch_case():
    """
    Fetch full case text from download_url.

    Request body:
    {
        "download_url": "https://www.supremecourt.gov/opinions/...",
        "case_name": "Brown v. Board of Education",
        "chunk_text": "text from the chunk for context"
    }

    Response:
    {
        "case_name": "...",
        "full_text": "...",
        "chunk_text": "...",
        "success": true
    }
    """
    try:
        data = request.get_json()

        if not data or 'download_url' not in data:
            return jsonify({
                "error": "Missing 'download_url' field in request body"
            }), 400

        download_url = data['download_url']
        case_name = data.get('case_name', 'Unknown Case')
        chunk_text = data.get('chunk_text', '')

        logger.info(f"Fetching case from: {download_url}")

        # Fetch the PDF
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()

        # Extract text from PDF
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")

        return jsonify({
            "case_name": case_name,
            "full_text": full_text,
            "chunk_text": chunk_text,
            "success": True,
            "text_length": len(full_text)
        }), 200

    except requests.RequestException as e:
        logger.error(f"Failed to fetch PDF: {e}")
        return jsonify({
            "error": f"Failed to fetch PDF: {str(e)}",
            "success": False
        }), 500
    except Exception as e:
        logger.error(f"Case fetch error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/case/summarize-stream', methods=['POST'])
def summarize_case_stream():
    """
    Stream GPT summary of full case text character by character using SSE.

    Request body:
    {
        "case_name": "Brown v. Board of Education",
        "full_text": "...",
        "chunk_text": "...",
        "user_question": "What is the holding in this case?"
    }

    Response: Server-Sent Events stream
    """
    try:
        data = request.get_json()

        if not data or 'full_text' not in data:
            return jsonify({
                "error": "Missing 'full_text' field in request body"
            }), 400

        case_name = data.get('case_name', 'Unknown Case')
        full_text = data['full_text']
        chunk_text = data.get('chunk_text', '')
        user_question = data.get('user_question', '')

        if not OPENAI_AVAILABLE or not openai_client:
            return jsonify({
                "error": "OpenAI API not available. Please set OPENAI_API_KEY."
            }), 503

        logger.info(f"Streaming summary for case: {case_name}")

        # Truncate full_text if too long (GPT token limits)
        max_chars = 12000  # Roughly 3000 tokens
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[Document truncated due to length...]"

        system_prompt = """You are a legal research assistant. Provide a comprehensive summary of the legal case that answers the user's question.

Guidelines:
- Start with the case name and basic information (court, date, parties)
- Explain the key facts of the case
- Describe the legal issues presented
- Explain the court's holding and reasoning
- Mention any important concurrences or dissents
- Keep the summary well-structured and professional
- Aim for 300-500 words for a complete analysis"""

        user_prompt = f"""Case Name: {case_name}

User's Question: {user_question}

Relevant Passage from Search:
{chunk_text[:500]}

Full Case Text:
{full_text}

Please provide a comprehensive summary that answers the user's question based on this case."""

        def generate():
            try:
                # Stream from OpenAI
                stream = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        # Send as SSE format
                        yield f"data: {content}\n\n"

                # Send completion signal
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        logger.error(f"Summarize stream error: {e}")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    logger.info(f"Starting Legal RAG Chatbot Service on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
