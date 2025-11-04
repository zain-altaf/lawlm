FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data_ingestion.py opinion_utills.py ./

# Environment variable for API key (pass via docker run -e)
ENV CASELAW_API_KEY="your_caselaw_api_key_here"

# Create output directory for persisting data
RUN mkdir -p /app/output
VOLUME /app/output

CMD ["python", "data_ingestion.py"]