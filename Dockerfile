# The Energiewende agent API.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    PYTHONPATH=/app/src

# Packages first, so Docker can reuse this layer when only the code changes.
# PyTorch from the CPU-only index: no GPU libraries, a much smaller image.
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Download the embedding model at build time, so the container works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
ENV HF_HUB_OFFLINE=1

COPY src/ src/

# The search index is not in the image. Mount it: -v ./data:/app/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "energiewende.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
