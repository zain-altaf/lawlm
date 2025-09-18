# Legal Document Processing Pipeline

A production-ready legal document processing pipeline that ingests legal documents from the CourtListener API, processes them into chunks, creates vector embeddings, and stores them in Qdrant (Local or Cloud) for hybrid search. Features full Airflow orchestration with Redis-based distributed rate limiting (5000 API calls/hour) and both direct execution and DAG-based processing modes.

## 🎯 Overview

This pipeline provides core functionality for legal document processing and retrieval:

- **Data Ingestion**: Fetches legal cases from CourtListener API with Redis-based rate limiting (5000 calls/hour)
- **Text Processing**: Uses RecursiveCharacterTextSplitter for chunking documents with boundary and text overlap across chunks
- **Vector Processing**: Creates embeddings using BGE models (default is BAAI/bge-small-en-v1.5) and BM25 for sparse embeddings
- **Storage**: Qdrant vector database (local or cloud) with hybrid search capabilities
- **Orchestration**: Full Airflow DAG execution with task dependencies and monitoring
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
├── manage_qdrant.sh            # Qdrant Docker management script
├── run_airflow.sh              # Airflow management script with Redis integration
├── airflow/                    # Airflow orchestration infrastructure
│   ├── dags/
│   │   └── courtlistener_pipeline_dag.py  # Main DAG implementation
│   ├── hooks/
│   │   └── redis_rate_limit_hook.py       # Redis hook for distributed rate limiting
│   ├── airflow.cfg             # Airflow configuration
│   └── webserver_config.py     # Webserver configuration
├── data/                       # Working directory for pipeline files (created after main.py is run)
├── qdrant_storage/             # Local Qdrant storage (created if Qdrant is run locally and after main.py is run)
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10** with pip (preferrably in an isolated conda environment)
2. **CourtListener API key** ([get one here](https://www.courtlistener.com/api/))
3. **Docker** (for Redis and local Qdrant instances)
4. **Qdrant API** (if using cloud deployment, otherwise will save locally)

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

Configure your `.env` file with the required credentials:

**Required for Data Ingestion:**
- `CASELAW_API_KEY`: CourtListener API key for legal document access

**For Qdrant Cloud Deployment:**
- `QDRANT_URL`: Cluster URL (e.g., https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333)
- `QDRANT_API_KEY`: Cloud API key
- `QDRANT_CLUSTER_NAME`: Cluster name
- `USE_CLOUD`: Set to "true" for cloud deployment, "false" for local

**For RAG Queries:**
- `OPENAI_API_KEY`: Required for legal_rag_query.py to generate responses

**For Redis (Airflow Rate Limiting):**
- `AIRFLOW_CONN_REDIS_DEFAULT`: Fallback connection string (redis://localhost:6379/0)

### Running this Repository:

#### Option 1: Airflow Orchestration (Recommended)

```bash
# Start Airflow services and Redis (local)
./run_airflow.sh                 # Start/ensure services are running
./manage_qdrant.sh start         # If you're using local qdrant

# Reset Airflow services, reset Redis and delete Database
./run_airflow.sh --reset

# Stop Airflow services and Redis (database remains)
./run_airflow.sh --stop

# Start Airflow services and Redis (cloud)
QDRANT_URL: Cluster URL (e.g., https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333)
QDRANT_API_KEY: Cloud API key
QDRANT_CLUSTER_NAME: Cluster name
USE_CLOUD: true

# run airflow
./run_airflow.sh

# Access Airflow UI
# Navigate to http://localhost:8080 (admin/admin)
```

#### Option 2: Manual Execution

```bash
# Enable qdrant usage locally via Docker
./manage_qdrant.sh start # Starts local instance
```

```bash
# Run complete pipeline (NOTE: schema for scotus works. Not tested for other courts at this time)
python main.py --court scotus --num-dockets 5

# Check pipeline status (including key configurations)
python main.py --status
```

##### Useful qdrant docker commands
```bash
# Other useful ./manage_qdrant.sh commands
./manage_qdrant.sh stop # Stops the local instance

./manage_qdrant.sh restart # Restarts local instance

./manage_qdrant.sh status # Checks on Docker status and runs health check

./manage_qdrant.sh logs # Displays the logs from the container. Press Ctrl + C to exit

./manage_qdrant.sh clean # Deletes all data from /qdrant_storage

./manage_qdrant.sh help # Displays the list of commands and helpful information
```

#### Using Qdrant cloud

Ensure you have a working QDRANT_API_KEY, QDRANT_URL and QDRANT_CLUSTER_NAME and make sure you set USE_CLOUD=true. This will ensure the switch from local to cloud upload of text chunks.

##### Running legal_rag_query.py

Prerequisites: 
1. Ensure you have a working OPENAI_API_KEY in .env.
2. Ensure you have a working QDRANT_API_KEY, QDRANT_URL and QDRANT_CLUSTER_NAME. Make sure USE_CLOUD=true 
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
- ~~Implement cron jobs/orchestrators to automate ingestion cycles daily~~
- Allow users to use different embedding models and llms for vector embedding and querying
- Add test suite for pipeline components
- Use CourtListeners webhook to pull newer cases and auto update qdrant
- Enable batch processing when large amounts (>20) dockets are requested 

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request