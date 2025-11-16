# building data ingestion using the base image
FROM lawlm-base:latest
WORKDIR /app
COPY data-ingestion/ ./
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "data_extraction.py"]