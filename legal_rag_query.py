#!/usr/bin/env python3
"""
Legal RAG Query System

A retrieval-augmented generation system for legal document queries.
Combines hybrid search (semantic + keyword) with GPT API for concise summaries.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_processor import EnhancedVectorProcessor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI client setup
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False
except Exception as e:
    logger.warning(f"OpenAI setup failed: {e}")
    OPENAI_AVAILABLE = False


class LegalRAGSystem:
    """Legal Retrieval-Augmented Generation System."""
    
    def __init__(self, 
                 collection_name: str = "caselaw-chunks",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 openai_model: str = "gpt-4o-mini",
                 max_results: int = 5):
        """
        Initialize the Legal RAG system.
        
        Args:
            collection_name: Name of the hybrid Qdrant collection
            embedding_model: Embedding model for vector search
            openai_model: OpenAI model for text generation
            max_results: Maximum number of search results to consider
        """
        self.collection_name = collection_name
        self.openai_model = openai_model
        self.max_results = max_results
        
        # Initialize vector processor for hybrid search
        # Let EnhancedVectorProcessor handle URL logic based on USE_CLOUD flag
        self.vector_processor = EnhancedVectorProcessor(
            model_name=embedding_model,
            collection_name=collection_name
        )
        
        # Check if collection exists
        if not self.vector_processor.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' not found. Run the pipeline first to create it.")
        
        logger.info(f"✅ Legal RAG System initialized")
        logger.info(f"   Collection: {collection_name}")
        logger.info(f"   Embedding model: {embedding_model}")
        logger.info(f"   OpenAI model: {openai_model}")
        
        # Get collection info
        info = self.vector_processor.client.get_collection(collection_name)
        logger.info(f"   Available documents: {info.points_count} chunks")
    
    def search_legal_documents(self, 
                              query: str, 
                              score_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search legal documents using hybrid search (semantic + keyword).
        
        Args:
            query: User's legal query
            score_threshold: Minimum relevance score
            
        Returns:
            List of relevant document chunks with metadata
        """
        logger.info(f"🔍 Searching for: '{query}' (hybrid search)")
        
        try:
            results = self.vector_processor.hybrid_search(
                query=query,
                collection_name=self.collection_name,
                limit=self.max_results,
                score_threshold=score_threshold
            )
            
            logger.info(f"✅ Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into context for GPT."""
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
            
            # Clean up the text
            text_preview = text.strip()[:500]  # First 500 characters
            
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
        """Generate a concise summary using OpenAI API."""
        if not OPENAI_AVAILABLE:
            return f"OpenAI not available. Found {len(context.split('Document'))-1} relevant documents for query: '{query}'"
        
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
                temperature=0.1,  # Low temperature for factual responses
                top_p=0.95
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated {len(summary.split())} word summary")
            return summary
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return f"Error generating summary. Found relevant information about: {query}"
    
    def query(self, 
             question: str, 
             score_threshold: float = 0.4,
             show_sources: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query: search + generate summary.
        
        Args:
            question: Legal question to answer
            score_threshold: Minimum relevance score
            show_sources: Whether to include source information
            
        Returns:
            Dictionary with summary, sources, and metadata
        """
        start_time = datetime.now()
        
        # Step 1: Search relevant documents
        search_results = self.search_legal_documents(
            query=question,
            score_threshold=score_threshold
        )
        
        if not search_results:
            return {
                "question": question,
                "summary": "No relevant legal documents found for this query. Try rephrasing your question or using different keywords.",
                "sources": [],
                "search_type": "hybrid_rrf",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "documents_found": 0
            }
        
        # Step 2: Format context for GPT
        context = self.format_search_results(search_results)
        
        # Step 3: Generate summary
        summary = self.generate_summary(question, context)
        
        # Step 4: Prepare sources information
        sources = []
        if show_sources:
            for result in search_results:
                payload = result['payload']
                source = {
                    "case_name": payload.get('case_name', 'Unknown Case'),
                    "court": payload.get('court_id', 'Unknown'),
                    "date_filed": payload.get('date_filed', 'Unknown'),
                    "opinion_type": payload.get('opinion_type', ''),
                    "relevance_score": result.get('score', 0),
                    "chunk_id": payload.get('chunk_id', '')
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


def main():
    """Command line interface for the Legal RAG system."""
    parser = argparse.ArgumentParser(description="Legal RAG Query System")
    parser.add_argument("--collection", default="caselaw-chunks",
                       help="Qdrant collection name")
    parser.add_argument("--model", default="gpt-4o-mini", 
                       help="OpenAI model to use")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive query session")
    parser.add_argument("--query", "-q", help="Single query to process")
    parser.add_argument("--max-results", type=int, default=5,
                       help="Maximum search results to consider")
    parser.add_argument("--score-threshold", type=float, default=0.4,
                       help="Minimum relevance score threshold")
    
    args = parser.parse_args()
    
    # Check required environment variables
    if not os.getenv("QDRANT_URL"):
        print("❌ Error: QDRANT_URL environment variable not set")
        return 1
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set. Summaries will be unavailable.")
    
    try:
        # Initialize RAG system
        rag_system = LegalRAGSystem(
            collection_name=args.collection,
            openai_model=args.model,
            max_results=args.max_results
        )
        
        if args.query:
            # Single query mode
            print(f"\n🔍 Legal Query: {args.query}")
            print("=" * 60)
            
            result = rag_system.query(
                question=args.query,
                score_threshold=args.score_threshold
            )
            
            print(f"\n📝 Summary ({len(result['summary'].split())} words):")
            print(result['summary'])
            
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])} documents):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['case_name']} ({source['court'].upper()}, {source['date_filed']}) - Score: {source['relevance_score']:.3f}")
            
            print(f"\n⏱️ Processing time: {result['processing_time']:.2f}s")
            
        elif args.interactive:
            # Interactive mode
            print("\n🏛️ Legal RAG Query System - Interactive Mode")
            print("Ask legal questions and get AI-powered summaries based on case law.")
            print("Type 'quit' or 'exit' to stop.")
            print("=" * 60)
            
            while True:
                try:
                    query = input("\n📋 Enter your legal question: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not query:
                        continue
                    
                    print(f"\n🔍 Searching and analyzing...")
                    
                    result = rag_system.query(
                        question=query,
                        score_threshold=args.score_threshold
                    )
                    
                    print(f"\n📝 Answer ({len(result['summary'].split())} words):")
                    print("-" * 40)
                    print(result['summary'])
                    
                    if result['sources']:
                        print(f"\n📚 Based on {len(result['sources'])} legal documents:")
                        for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                            print(f"  • {source['case_name']} ({source['court'].upper()}, {source['date_filed']})")
                    
                    print(f"\n⏱️ {result['processing_time']:.1f}s | {result['search_type']} search | {result['documents_found']} docs found")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        else:
            print("Please use --query for single question or --interactive for session mode")
            return 1
            
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())