FROM python:3.9-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with increased timeout
# Use PyTorch CPU-only index for faster downloads
RUN pip install --no-cache-dir --timeout=1200 --retries=5 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy all Python modules required for the pipeline
COPY pipeline_runner.py .
COPY config.py .
COPY hybrid_indexer.py .
COPY legal_rag_query.py .

# Copy .env template (users will need to mount their actual .env at runtime)
COPY .env.template .env.template

# Create necessary directories for data
RUN mkdir -p /app/data

# Use ENTRYPOINT for the base command and CMD for default arguments
# This allows users to: 
# 1. Run default: docker run image
# 2. Override args: docker run image --court scotus --num-dockets 5
ENTRYPOINT ["python", "pipeline_runner.py"]
CMD ["--court", "scotus", "--num-dockets", "50"]