# Legal Document Processing Pipeline

A robust legal document processing pipeline that ingests legal documents from the CourtListener API, processes them into chunks, creates vector embeddings, and stores them in Qdrant for semantic search. The system uses Legal BERT for understanding legal text and BGE embeddings for vector representations.

## 🎯 Overview

This pipeline provides core functionality for legal document processing and retrieval:

- **Data Ingestion**: Fetches legal cases from CourtListener API with batch processing support
- **Text Processing**: Uses RecursiveCharacterTextSplitter for chunking documents with enhanced boundary handling
- **Vector Processing**: Creates embeddings using BGE models (BAAI/bge-small-en-v1.5)
- **Storage**: Qdrant vector database (local or cloud) with semantic search capabilities
- **Query Interface**: RAG-based legal document retrieval system
- **Incremental Processing**: Memory-efficient processing that prevents memory buildup

## 🏗️ Architecture

```
CourtListener API → Data Ingestion → Text Chunking → Vector Processing → Qdrant Storage
```

### Pipeline Flow:
1. **Data Ingestion**: Fetch legal documents from CourtListener API
2. **Text Chunking**: Break documents into coherent chunks using RecursiveCharacterTextSplitter. It prioritizes chunking paragraphs and sentences that are complete along with some overlapping between chunks of text.
3. **Vector Processing**: Create semantic embeddings using BGE model (this can be modified to other embedding models if the user chooses)
4. **Storage**: Store vector representations of chunks along with payloads with descriptive fields in Qdrant vector database with hybrid search capabilities

## 📁 File Structure

```
lawlm/
├── README.md                   # Main documentation
├── requirements.txt            # Python dependencies
├── pipeline_runner.py          # Main pipeline orchestrator with data ingestion
├── config.py                   # Configuration management
├── hybrid_indexer.py           # Vector processing and text extraction
├── legal_rag_query.py          # RAG system for querying the vector database
├── manage_qdrant.sh            # Qdrant management script
├── data/                       # Working directory for pipeline files
├── qdrant_storage/             # Local Qdrant storage
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** with pip
2. **CourtListener API key** ([get one here](https://www.courtlistener.com/api/))
3. **Qdrant** (local or cloud deployment)

### Installation

**Note**: It is strongly recommended that you use a package manager (such as conda, or venv) when working with dependencies. You can install miniconda, a minimal installer for Conda here: https://www.anaconda.com/docs/getting-started/miniconda/install


```bash
# Create and activate a miniconda environment
conda create -n lawlmenv -y python=3.9
conda activate lawlmenv

# Clone repository
git clone https://github.com/zain-altaf/lawlm.git
cd lawlm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys
```

### Environment Configuration

Configure your `.env` file with the required credentials:

```bash
# Required
CASELAW_API_KEY=your_courtlistener_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here  # if using cloud
```

### Basic Usage

```bash
# Run complete pipeline (incremental processing)
python pipeline_runner.py --court scotus --num-dockets 5

# Process larger dataset
python pipeline_runner.py --court scotus --num-dockets 50

# Check pipeline status
python pipeline_runner.py --status
```

## 📋 Commands

### Pipeline Operations

```bash
# Run complete pipeline (ingest, chunk, vectorize) - uses incremental processing
python pipeline_runner.py --court scotus --num-dockets 5

# Run with larger dataset
python pipeline_runner.py --court scotus --num-dockets 50

# Check pipeline status
python pipeline_runner.py --status
```

### Individual Components

```bash
# Query the vector database
python legal_rag_query.py "What are the requirements for due process?"

# Run hybrid indexing (standalone mode)
python hybrid_indexer.py chunks_file.json
```

### Configuration

```bash
# Validate configuration
python config.py --validate

# View configuration summary
python config.py --summary
```

## 📂 Key Files and Their Purposes

- **pipeline_runner.py**: Main orchestrator that handles data ingestion from CourtListener API, text chunking, and calls vector processing
- **hybrid_indexer.py**: Contains all text processing functions (entity extraction, citation extraction), vector embedding creation, and Qdrant upload
- **legal_rag_query.py**: RAG system for querying the vector database with natural language
- **config.py**: Central configuration management for all pipeline components

## 🔧 Environment Variables

Required in `.env`:
- `CASELAW_API_KEY`: CourtListener API key
- `QDRANT_URL`: Qdrant server URL (local or cloud)
- `QDRANT_API_KEY`: Qdrant API key (for cloud deployments)

## 🔧 Important Implementation Details

1. **Chunking Strategy**: Uses RecursiveCharacterTextSplitter with enhanced separators that prioritize complete paragraphs and sentences. Default chunk size is 1536 characters with 300-character overlap for context continuity.

2. **Text Boundary Handling**: Advanced overlap fixing ensures chunks start and end at natural sentence boundaries, eliminating fragments like ". So, tak" at chunk beginnings.

3. **Vector Storage**: Supports both local and cloud Qdrant deployments. Cloud recommended for persistence.

4. **Processing Mode**: Uses incremental processing by default - processes each docket/batch completely (fetch → chunk → vectorize → upload) before moving to the next. Prevents memory buildup and ensures partial results are saved.

5. **Cursor-Based Pagination**: Uses CourtListener API's cursor pagination instead of page numbers, enabling reliable access to 500k+ historical SCOTUS dockets without ordering issues.

6. **Smart Deduplication**: Automatically detects and skips duplicate documents by docket number, with smart pagination that starts from unprocessed content.

7. **Error Handling**: Comprehensive error handling with detailed logging. Saves progress after each docket/batch.

8. **Resource Management**: Memory-efficient processing with configurable batch sizes and garbage collection.

## 💡 Common Tasks

### Process New Legal Documents
1. Ensure environment variables are set in `.env`
2. Run: `python pipeline_runner.py --court <court_id> --num-dockets <number>`
3. Monitor progress in console output
4. Query processed documents: `python legal_rag_query.py "<your legal question>"`

### Debug Issues
- Check configuration: `python config.py --validate`
- Enable debug logging: `export LOG_LEVEL=DEBUG`
- Check Qdrant connection: `curl http://localhost:6333/health` (or cloud URL)

### Handle Large Datasets
- Uses incremental processing by default (memory efficient)
- All processing is handled automatically with smart pagination

## 🔧 Recent Changes (2025-08-28)

### Text Chunking Improvements
Major improvements to text chunking quality and boundary handling:

1. **Enhanced Separator Hierarchy**: Updated RecursiveCharacterTextSplitter to prioritize:
   - Paragraph breaks (\n\n) - highest priority
   - Sentence endings (. ? !) - clean sentence breaks  
   - Line breaks (\n) - preserve structure
   - Word boundaries ( ) - natural fallback
   - Character level - last resort

2. **Chunk Overlap Boundary Fixing**: Added post-processing to fix overlap fragments:
   - Detects and removes leading punctuation fragments like ". So, tak"
   - Ensures chunks start at proper sentence boundaries with capital letters
   - Maintains semantic coherence across chunk boundaries
   - All chunks now start cleanly (100% success rate in testing)

3. **Cursor-Based API Pagination**: Replaced broken page-based pagination with cursor-based:
   - Enables access to all 500k+ historical SCOTUS dockets
   - More reliable than calculated page skipping
   - Proper handling of API ordering inconsistencies
   - Smart consecutive empty page detection

### File Structure Changes
- **Moved hybrid_indexer.py to root directory**: Simplified import structure for easier maintenance
- **Updated all references**: Fixed imports in pipeline_runner.py and legal_rag_query.py

## ℹ️ Notes

- No test suite currently exists in the codebase
- No linting or type checking configuration found
- Default models: Legal BERT for understanding, BGE for embeddings
- Incremental processing mode is recommended for datasets larger than 20 dockets
- Uses RecursiveCharacterTextSplitter with enhanced separators for improved chunking

## 🔄 Open Tasks / TODO

Tasks to be implemented in future iterations:

### High Priority
- [ ] **Implement Hybrid Search**: Enable `create_hybrid_index()` and `hybrid_search()` methods in pipeline
  - Currently only dense vectors are created, sparse vectors are not utilized
  - Need to modify pipeline_runner.py to call `create_hybrid_index()` instead of `process_and_upload_batch()`
  - Files: `hybrid_indexer.py:713`, `hybrid_indexer.py:944`

### Medium Priority
- [ ] Add test suite for pipeline components
- [ ] Implement linting and type checking configuration
- [ ] Add web interface for document upload and search

### Low Priority
- [ ] Add support for additional embedding models
- [ ] Implement document versioning and updates
- [ ] Add support for more court systems beyond SCOTUS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Note**: This pipeline uses incremental processing by default for memory efficiency. The system processes each docket/batch completely (fetch → chunk → vectorize → upload) before moving to the next to prevent memory buildup and ensure partial results are saved.