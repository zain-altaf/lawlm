# Legal Document Processing Pipeline

A legal document processing pipeline that ingests legal documents from the CourtListener API, processes them into chunks, creates vector embeddings, and stores them in Qdrant locally for hybrid search. Users can then ask it questions using a chat bot to get: 1. a RAG based answer from an LLM, and 2. top 3 cases based on relevance between query and qdrant data points.

## Overview

This pipeline provides core functionality for legal document processing and retrieval:

- **Data Ingestion**: Fetches legal cases from CourtListener API
- **Text Processing**: Uses RecursiveCharacterTextSplitter for chunking documents with boundary and text overlap across chunks
- **Vector Processing**: Creates embeddings using BAAI/bge-small-en-v1.5 and BM25 for sparse embeddings
- **Storage**: local Qdrant vector database
- **Query Interface**: RAG-based legal document retrieval system available in chatbot service

## 📁 File Structure

```
lawlm/
├── README.md                   # Main documentation
├── docker-compose.yml          # Docker orchestration configuration
├── .env.template               # Environment variables template (API keys only)
├── config.yml                  # Centralized configuration (chunking, models, ports, etc.)
├── Dockerfile.base             # Shared base image with common dependencies
├── requirements-base.txt       # Shared Python dependencies (PyTorch, transformers, etc.)
├── data-ingestion/             # Data ingestion microservice
│   ├── Dockerfile              # Container definition (extends base image)
│   ├── requirements.txt        # Service-specific Python dependencies
│   ├── data_extraction.py      # Main ingestion pipeline
│   └── opinion_utills.py       # Text processing utilities
├── chatbot/                    # Web-based RAG query service
│   ├── Dockerfile              # Container definition (extends base image)
│   ├── requirements.txt        # Service-specific Python dependencies
│   ├── app.py                  # Flask REST API
│   └── static/
│       └── index.html          # Web interface for legal search
└── qdrant_storage/             # Local Qdrant data (auto-created by Docker)
```

## Quick Start

### Prerequisites

1. **Docker** and **Docker Compose** ([install Docker](https://docs.docker.com/get-docker/))
2. **CourtListener API key** ([get one here](https://www.courtlistener.com/api/))
3. **OpenAI API key** (optional - only needed for AI-generated summaries via chatbot interface) ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone repository
git clone https://github.com/zain-altaf/lawlm.git
cd lawlm

# Set up environment variables
cp .env.template .env
# Edit .env and add your CourtListener API key (required)
# Optionally add your OpenAI API key for query summaries
# QDRANT_URL is already set to http://localhost:6333 for local use

# Build and start all services (Qdrant, data ingestion, chatbot)
docker compose up --build # the data ingestion service will, by default, process one page of the SCOTUS cases
```

This will start three services:
- **Qdrant** (vector database) on ports 6333 and 6334
- **Data ingestion** service that processes legal cases and stores them in Qdrant
- **Chatbot** web interface on http://localhost:5000

### Using the Application

Once all services are running, you can access the web interface:
   - Navigate to http://localhost:5000 in your browser
   - Enter your legal question in the search box
   - View AI-generated summaries with source cases and relevant passages
   - Click "View Full Case" to access original PDF documents

3. **Stopping Services**
   ```bash
   # Stop all services
   docker compose down

   # Stop and remove all data (including vector database)
   docker compose down -v
   ```

### Configuration

The application can be customized via two files:

1. **[.env](.env)** - Environment variables (sensitive data only):
   - `CASELAW_API_KEY`: CourtListener API key (required)
   - `QDRANT_URL`: Qdrant database URL (default: http://qdrant:6333 for Docker)
   - `OPENAI_API_KEY`: OpenAI API key for summaries (optional)

2. **[config.yml](config.yml)** - Application settings (non-sensitive):
   - **Chunking**: `chunk_size_chars` (1536), `overlap_chars` (300), `min_chunk_size_chars` (400)
   - **Vectorization**: `embedding_model` (BAAI/bge-small-en-v1.5), `vector_size` (384)
   - **Qdrant**: `collection_name` (caselaw-chunks-scotus)
   - **API**: `court` (scotus), `request_delay` (0.5s), `max_retries` (3)
   - **Services**: Ports for data-ingestion (5001) and chatbot (5000)
   - **RAG**: `openai_model` (gpt-4o-mini), `max_results` (3), `default_score_threshold` (0.4)

### Docker Architecture

The project uses an **optimized multi-stage Docker build** to minimize storage usage:

**Shared Base Image** ([Dockerfile.base](Dockerfile.base)):
- Contains all common dependencies (PyTorch, transformers, qdrant-client, etc.)
- Built once and shared by both services
- Size: ~5.86GB

**Service-Specific Images**:
- `data-ingestion`: Base + service-specific deps (~40MB)
- `chatbot`: Base + service-specific deps (~10MB)

**Storage Savings**: ~50% reduction compared to separate builds (5.91GB vs 11.77GB)

The base image is automatically built first via `docker compose build`, and both services extend it. Docker's layer caching ensures the base layers are shared on disk.

## 🔄 Open Tasks / TODO

Tasks to be implemented in future iterations:

- ~~Create a hybrid search method in pipeline for enhanced querying in RAG~~
- Add functionality around audio/oral arguments in the instance that clusters are not available for scraping of court cases
- Implement cron jobs/orchestrators to automate ingestion cycles daily (currently data_extraction.py only runs once at startup)
- Add comprehensive test suite for pipeline components (currently only have one for ensuring duplicate dockets have not been uploaded)
- Allow users to use different embedding models and llms for vector embedding and querying
- Use CourtListeners webhook to pull newer cases and auto update qdrant