# Hugging Face Space image: RAG evaluation API (in-memory Chroma + Groq).
# Listening on the Space's fixed port 7860.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

# Pre-warm the chroma ONNX embedding cache so the first ingest does not
# pull 79MB over a slow connection on every restart.
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()"

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4); sys.exit(0)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
