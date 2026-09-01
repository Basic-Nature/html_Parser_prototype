# syntax=docker/dockerfile:1

# ============================================================
# Election Pulse / BallotLens production container
#
# Keep Python 3.12 for the project's tested dependency ABI.
#
# Build strategy:
#   1. Build Python dependencies in a builder stage.
#   2. Carry only the finished virtual environment forward.
#   3. Keep compilers and development headers out of runtime.
#   4. Install only Chromium for Playwright.
#   5. Bake the required SentenceTransformer model for
#      offline Azure runtime use.
# ============================================================

FROM node:24.20.0-bookworm-slim AS frontend-builder

ENV PUPPETEER_SKIP_DOWNLOAD=true

WORKDIR /build/webapp/frontend/ballot-lens

COPY webapp/frontend/ballot-lens/package.json webapp/frontend/ballot-lens/package-lock.json ./

RUN npm ci --no-audit --no-fund

COPY webapp/frontend/ballot-lens ./

RUN npm run verify

# ============================================================
# Python dependency builder
# ============================================================

FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        gcc \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

COPY requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Preserve the currently required spaCy small English model.
RUN pip install --no-cache-dir \
    "spacy>=3.8,<3.9" \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Fail the image build immediately if core runtime libraries
# required by the parser are absent from requirements.txt.
RUN python - <<'PY'
import playwright
import sentence_transformers
import spacy
import torch

nlp = spacy.load("en_core_web_sm")

print("Playwright import: OK")
print("SentenceTransformers import: OK")
print("Torch import: OK")
print("spaCy model:", nlp.meta.get("name"), nlp.meta.get("version"))
PY

# ============================================================
# Bake SentenceTransformer model for offline production use
# ============================================================

RUN mkdir -p /models/sentence /models/hf-cache

ENV HUGGINGFACE_HUB_CACHE=/models/hf-cache \
    HF_HOME=/models/hf-cache

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

model.save(
    "/models/sentence/all-MiniLM-L6-v2"
)

print(
    "Saved SentenceTransformer model to "
    "/models/sentence/all-MiniLM-L6-v2"
)
PY

# ============================================================
# Bake NLTK resources required by application startup
# ============================================================

RUN mkdir -p /usr/local/share/nltk_data && python -c "import nltk; ok=nltk.download('stopwords', download_dir='/usr/local/share/nltk_data', quiet=True, raise_on_error=True); assert ok; print(nltk.data.find('corpora/stopwords', paths=['/usr/local/share/nltk_data']))"
# ============================================================
# Runtime stage
# ============================================================

FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_OCR=True \
    PLAYWRIGHT_BROWSERS_PATH=0 \
    NLTK_DATA=/usr/local/share/nltk_data \
    SENTENCE_TRANSFORMER_LOCAL_PATH=/models/sentence/all-MiniLM-L6-v2 \
    TRANSFORMERS_OFFLINE=1 \
    HUGGINGFACE_HUB_OFFLINE=1 \
    HUGGINGFACE_HUB_CACHE=/models/hf-cache \
    HF_HOME=/models/hf-cache

# Copy the finished Python environment and baked ML assets.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

ENV PATH="/opt/venv/bin:${PATH}"

# Runtime-only native dependencies.
#
# build-essential, gcc, and libpq-dev remain behind in the
# builder stage and are not copied into this image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        tesseract-ocr \
        poppler-utils \
        ghostscript \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libpq5 && \
    python -m playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/* \
           /root/.cache \
           /tmp/*

WORKDIR /app

COPY . .

COPY --from=frontend-builder /build/webapp/static/dist/ballot-lens-f2 /app/webapp/static/dist/ballot-lens-f2

EXPOSE 8000

CMD ["gunicorn", "--config", "gunicorn.conf.py", "webapp.Smart_Elections_Parser_Webapp:app"]
