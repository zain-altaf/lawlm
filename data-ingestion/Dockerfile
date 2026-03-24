FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data_extraction.py opinion_utills.py config.yml ./

# Create output directory for persisting data
RUN mkdir -p /app/output
VOLUME /app/output

CMD ["python", "data_extraction.py"]