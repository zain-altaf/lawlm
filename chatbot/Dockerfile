# building chatbot using the base image
FROM lawlm-base:latest
WORKDIR /app
COPY chatbot/ ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1
CMD ["python", "app.py"]