FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies with PyTorch CPU wheels
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --timeout=1200 --retries=5 \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

# Copy application files
COPY main.py config.py vector_processor.py legal_rag_query.py ./
COPY .env.template .env.template

# Create data directory
RUN mkdir -p /app/data

# Default entrypoint
ENTRYPOINT ["python", "main.py"]
CMD ["--court", "scotus", "--num-dockets", "5"]
