# Legal Document Processing Pipeline

A legal document processing pipeline that ingests legal documents from the CourtListener API, processes them into chunks, creates vector embeddings, and stores them in Qdrant (Local or Cloud) for hybrid search. The system uses BGE embeddings for dense vector representations and BM25 for sparse vector representations.

## 🎯 Overview

This pipeline provides core functionality for legal document processing and retrieval:

- **Data Ingestion**: Fetches legal cases from CourtListener API
- **Text Processing**: Uses RecursiveCharacterTextSplitter for chunking documents with boundary and text overlap across chunks.
- **Vector Processing**: Creates embeddings using BGE models (default is BAAI/bge-small-en-v1.5) and BM25 for sparse embeddings.
- **Storage**: Qdrant vector database (local or cloud) with hybrid search capabilities
- **Query Interface**: RAG-based legal document retrieval system available in legal_rag_query.py

## 📁 File Structure

```
lawlm/
├── README.md                   # Main documentation
├── requirements.txt            # Dependencies
├── main.py                     # Main pipeline orchestrator to ingest, process, and upload to vector database
├── config.py                   # Configuration management
├── vector_processor.py         # Text extraction, deduplication logic for existing records, and vector processing 
├── legal_rag_query.py          # RAG system for querying the vector database in command line
├── manage_qdrant.sh            # This helps you quickly start up qdrant locally via Docker
├── data/                       # Working directory for pipeline files (created after main.py is run)
├── qdrant_storage/             # Local Qdrant storage (created if Qdrant is run locally and after main.py is run)
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10** with pip (preferrably in an isolated conda environment)
2. **CourtListener API key** ([get one here](https://www.courtlistener.com/api/))
3. **Qdrant API** (if using cloud deployment, otherwise will save locally)

### Installation

**Note**: If you're not using the Docker implementation, ensure you use a package manager (such as conda, or venv) when working with dependencies. You can install miniconda, an installer for Conda here: https://www.anaconda.com/docs/getting-started/miniconda/install


```bash
# Create and activate a miniconda environment (ensure sure to use only v3.10)
conda create -n lawlmenv -y python=3.10
conda activate lawlmenv

# Clone repository
git clone https://github.com/zain-altaf/lawlm.git
cd lawlm

# Install dependencies (Note: only the CPU torch version is supported at this time)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys
```

### Environment Configuration

Configure your `.env` file with the required credentials (follow .env.template for more information)

### Running this Repository:

#### Using Qdrant locally with a Docker Container
```bash
# Enable qdrant usage locally via Docker
./manage_qdrant.sh start # Starts local instance

./manage_qdrant.sh stop # Stops the local instance

./manage_qdrant.sh restart # Restarts local instance

./manage_qdrant.sh status # Checks on Docker status and runs health check

./manage_qdrant.sh logs # Displays the logs from the container. Press Ctrl + C to exit

./manage_qdrant.sh clean # Deletes all data from /qdrant_storage

./manage_qdrant.sh help # Displays the list of commands and helpful information
```

```bash
# Run complete pipeline (NOTE: schema for scotus works. Not tested for other courts at this time)
python main.py --court scotus --num-dockets 5

# Check pipeline status (including key configurations)
python main.py --status
```
#### Using Qdrant cloud

Ensure you have a working QDRANT_API_KEY, QDRANT_URL and QDRANT_CLUSTER_NAME. This will ensure the switch from local to cloud upload. **NOTE**: Make sure you shut down any local qdrant with: 

```bash
./manage_qdrant.sh stop 
```

And then you can run:
```bash
python main.py --court scotus --num-dockets 5
```

And Qdrant cloud will recieve the text chunks.

##### Running legal_rag_query.py

Prerequisites: 
1. Ensure you have a working OPENAI_API_KEY in .env.
2. Ensure you have a working QDRANT_API_KEY, QDRANT_URL and QDRANT_CLUSTER_NAME. 
3. Ensure there are at least 5-10 dockets in Qdrant cloud for better semantic search prior to querying OPEN AI. 

```bash
# Query the vector database
python legal_rag_query.py --query "Can you tell me about the case Noem v. Vasquez Perdomo"
```

## 🔧 Important Implementation Details

1. **Chunking Strategy**: Uses RecursiveCharacterTextSplitter with enhanced separators. Default chunk size is 1536 characters with 300-character overlap for context continuity.

3. **Vector Storage**: Supports both local and cloud Qdrant deployments. Cloud recommended for persistence. Snapshot of local embeddings can be downloaded from qdrant for later local use or cloud upload.

4. **Processing Mode**: Uses incremental processing - processes each docket (fetch → chunk → vectorize → upload) before moving to the next.

5. **Cursor-Based Pagination**: Uses CourtListener API's cursor pagination, enabling reliable access to ~500k historical SCOTUS dockets without ordering issues.

## 🔄 Open Tasks / TODO

Tasks to be implemented in future iterations:

- ~~Create a hybrid search method in pipeline for enhanced querying in RAG~~
- Allow users to use different embedding models and llms for vector embedding and querying
- Add test suite for pipeline components
- Use CourtListeners webhook to pull newer cases and auto update qdrant
- Implement cron jobs/orchestrators to automate ingestion cycles daily
- Enable batch processing when large amounts (>20) dockets are requested 

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request