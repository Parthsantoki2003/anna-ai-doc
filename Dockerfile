FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Install minimal OS dependencies (curl for healthcheck, tesseract-ocr for OCR fallback)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man /var/cache/apt/* /var/log/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # Strip unnecessary heavy C++ headers, test suites, and caches (keep torch/bin for torch_shm_manager)
    && rm -rf /usr/local/lib/python3.11/site-packages/torch/include \
    && rm -rf /usr/local/lib/python3.11/site-packages/torch/test \
    && find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Copy pre-downloaded embedding model directly into the image (100% offline, zero HF download in Cloud Build)
COPY model_cache /app/model_cache

COPY main.py .
COPY manifest.json app.json ./
COPY bundle ./bundle

RUN mkdir -p /app/data /app/logs

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-1}"]
