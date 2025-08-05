# Legal Document Processing Pipeline

A robust, production-ready pipeline for processing legal documents from CourtListener API with semantic chunking and vector search capabilities. Designed for efficient ingestion, intelligent chunking, and scalable vector indexing of legal case data.

## 🎯 Overview

This pipeline focuses on the core functionality of **correctly ingesting, chunking, and pushing data** for legal document search. It provides:

- **Smart Data Ingestion**: Batch processing from CourtListener API with resume capability
- **Semantic Chunking**: Legal BERT-powered intelligent text segmentation
- **Vector Processing**: BGE embeddings optimized for legal domain
- **Batch Processing**: Robust handling of large document collections
- **Resource Optimization**: Memory-efficient processing with configurable limits

## 🏗️ Architecture

```
CourtListener API → Data Ingestion → Semantic Chunking → Vector Processing → Qdrant Storage
```

### Pipeline Steps:
1. **Data Ingestion**: Fetch legal documents from CourtListener API
2. **Semantic Chunking**: Break documents into coherent chunks using Legal BERT
3. **Vector Processing**: Create semantic embeddings using BGE models
4. **Storage**: Store vectors in Qdrant for efficient search

## 📁 File Structure

```
lawlm/
├── README.md                   # Main documentation
├── requirements.txt            # Python dependencies
├── pipeline_runner.py          # Main pipeline orchestrator
├── config.py                  # Configuration management
├── monitor.py                 # System monitoring and health checks
├── batch_utils.py             # Batch processing utilities
├── data_ingestion.py          # CourtListener API integration
├── processing/
│   ├── __init__.py            # Processing module exports
│   ├── smart_chunking.py      # Semantic chunking with Legal BERT
│   └── vector_processor.py    # Vector processing with BGE embeddings
├── data/                      # Working directory for pipeline files
├── qdrant_storage/           # Qdrant vector database storage
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Qdrant server** running locally or remotely
3. **CourtListener API key** (set as environment variable)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd lawlm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your CASELAW_API_KEY and QDRANT_URL
```

### Basic Usage

```bash
# Run complete pipeline with defaults (5 Supreme Court cases)
python pipeline_runner.py --court scotus --num-dockets 5

# Process with batch processing (recommended for larger datasets)
python pipeline_runner.py --court scotus --num-dockets 20 --batch-size 5

# Check pipeline status
python pipeline_runner.py --status
```

## 📊 Configuration

The pipeline supports comprehensive configuration through:

### 1. Environment Variables
```bash
CASELAW_API_KEY=your_api_key_here
QDRANT_URL=http://localhost:6333
```

### 2. Configuration Files
```bash
# Create default configuration
python config.py --create-default

# Validate configuration
python config.py --validate

# View configuration summary
python config.py --summary
```

### 3. Command Line Options

#### Data Ingestion
- `--court`: Court identifier (default: scotus)
- `--num-dockets`: Number of dockets to fetch (default: 5)
- `--batch-size`: Enable batch processing with specified size

#### Processing Options
- `--working-dir`: Working directory for files (default: data)
- `--chunk-size`: Target chunk size in tokens (default: 384)
- `--embedding-model`: BGE model for embeddings (default: BAAI/bge-small-en-v1.5)

#### Advanced Options
- `--enable-hybrid`: Enable hybrid search (resource intensive, disabled by default)
- `--resume`: Resume interrupted batch processing
- `--job-id`: Custom job identifier for batch processing

## 🔧 Key Features

### Batch Processing
- **Smart Division**: Handles non-evenly divisible batch sizes
- **Resume Capability**: Continue from interruption points
- **Progress Tracking**: Real-time status and completion metrics
- **Error Recovery**: Robust handling of failed batches

### Memory Optimization
- **Lazy Loading**: Components loaded only when needed
- **Garbage Collection**: Aggressive cleanup during processing
- **GPU Memory Management**: CUDA memory optimization
- **Resource Monitoring**: Real-time memory usage tracking

### Quality Assurance
- **Chunk Validation**: 6-metric quality assessment
- **Legal Content Detection**: Ensures legal domain relevance
- **Input Validation**: Comprehensive parameter checking
- **Error Handling**: Detailed logging and recovery mechanisms

## 💡 Usage Examples

### Basic Document Processing
```bash
# Process 10 Supreme Court cases
python pipeline_runner.py --court scotus --num-dockets 10

# Process with custom chunk size
python pipeline_runner.py --court scotus --num-dockets 5 --chunk-size 512
```

### Batch Processing (Recommended)
```bash
# Process 50 cases in batches of 10
python pipeline_runner.py --court scotus --num-dockets 50 --batch-size 10

# Resume interrupted job
python pipeline_runner.py --resume --job-id scotus_50dockets_20250805_123456

# Custom batch settings
python pipeline_runner.py --court ca9 --num-dockets 30 --batch-size 5 --job-id my_ca9_job
```

### Individual Components
```bash
# Data ingestion only
python data_ingestion.py --court scotus --num_dockets 5

# Semantic chunking with custom parameters
python processing/smart_chunking.py data/raw_cases.json --quality_threshold 0.4

# Vector processing
python processing/vector_processor.py data/chunks.json

# System monitoring
python monitor.py --health
python monitor.py --watch 10  # Monitor every 10 seconds
```

## 🔍 Monitoring & Health Checks

### System Health
```bash
# Check overall system health
python monitor.py --health

# Continuous monitoring
python monitor.py --watch 5  # Check every 5 seconds

# Memory usage monitoring
python monitor.py --memory
```

### Pipeline Status
```bash
# Check pipeline status and files
python pipeline_runner.py --status

# View configuration summary
python config.py --summary
```

## ⚙️ Configuration Reference

### Data Ingestion
- `api_key`: CourtListener API key
- `timeout_seconds`: Request timeout (default: 30)
- `min_text_length`: Minimum text length for processing (default: 100)

### Semantic Chunking
- `model_name`: Legal BERT model (default: nlpaueb/legal-bert-base-uncased)
- `target_chunk_size`: Target tokens per chunk (default: 384)
- `quality_threshold`: Minimum quality score (default: 0.3)
- `clustering_threshold`: Semantic similarity threshold (default: 0.25)

### Vector Processing
- `embedding_model`: BGE model for embeddings (default: BAAI/bge-small-en-v1.5)
- `batch_size`: Processing batch size (default: 50)
- `collection_name_vector`: Qdrant collection name

### Batch Processing
- `enable_batch_processing`: Enable batch mode (default: true)
- `default_batch_size`: Default batch size (default: 5)
- `max_batch_size`: Maximum allowed batch size (default: 10)
- `checkpoint_interval`: Progress logging frequency (default: 100)

## 🐛 Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Reduce batch sizes
python pipeline_runner.py --batch-size 3

# Monitor memory usage
python monitor.py --memory
```

#### API Rate Limits
```bash
# Use smaller batch sizes and add delays
python pipeline_runner.py --batch-size 2
```

#### Qdrant Connection Issues
```bash
# Check Qdrant server status
curl http://localhost:6333/health

# Validate configuration
python config.py --validate
```

#### Interrupted Processing
```bash
# Resume from last checkpoint
python pipeline_runner.py --resume --job-id <your-job-id>
```

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python pipeline_runner.py --court scotus --num-dockets 5
```

## 🔬 Technical Details

### Models Used
- **Legal BERT**: `nlpaueb/legal-bert-base-uncased` for semantic understanding
- **BGE Embeddings**: `BAAI/bge-small-en-v1.5` for vector representations
- **Vector Dimension**: 384 (optimized for legal content)

### Processing Pipeline
1. **Text Extraction**: HTML/XML parsing with BeautifulSoup
2. **Legal Entity Recognition**: Citations, judges, parties, statutes
3. **Semantic Clustering**: Agglomerative clustering with cosine similarity
4. **Quality Validation**: 6-metric assessment including legal content detection
5. **Vector Generation**: Context-enhanced embeddings with BGE optimization

### Performance Characteristics
- **Throughput**: ~5-10 documents/minute (depends on document size)
- **Memory Usage**: ~2-4GB RAM for typical workloads
- **Storage**: ~1.5KB per chunk in Qdrant
- **Scalability**: Tested with 1000+ documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[License information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review configuration with `python config.py --validate`
3. Check system health with `python monitor.py --health`
4. Create an issue in the repository

---

**Note**: This pipeline is optimized for vector-only processing by default. Hybrid search capabilities are available but disabled due to resource requirements. Use `--enable-hybrid` flag if needed.