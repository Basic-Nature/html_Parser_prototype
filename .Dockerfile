# Use a stable, public base
FROM python:3.12-slim-bookworm

# System deps for OCR and Playwright
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    tesseract-ocr poppler-utils \
    libpq-dev build-essential gcc \
  && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV ENABLE_OCR=True
# Store Playwright browsers in the image filesystem (also set in App Settings)
ENV PLAYWRIGHT_BROWSERS_PATH=0

# Install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Ensure spaCy and the small English model are present even if not in requirements.txt
    pip install --no-cache-dir "spacy>=3.8,<3.9" \
      "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" && \
    # Install Playwright and browsers
    pip install --no-cache-dir playwright && \
    python -m playwright install --with-deps && \
    # Build-time verification that spaCy model is loadable
    python - <<'PY'
import spacy
nlp = spacy.load("en_core_web_sm")
print("spaCy model ready:", nlp.meta.get("version"))
PY

# --- Bake SentenceTransformer into the image (build-time) ---
# Create model/cache dirs and set HF caches (safe pre-download)
RUN mkdir -p /models/sentence /models/hf-cache
ENV HUGGINGFACE_HUB_CACHE=/models/hf-cache
ENV HF_HOME=/models/hf-cache

# Ensure sentence-transformers is available (if not already from requirements.txt)
RUN pip install --no-cache-dir "sentence-transformers>=2.7.0"

# Pre-download and vendor the model into the image
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
m.save("/models/sentence/all-MiniLM-L6-v2")
print("Saved ST model to /models/sentence/all-MiniLM-L6-v2")
PY

# Point app to the baked model and prefer offline at runtime
ENV SENTENCE_TRANSFORMER_LOCAL_PATH=/models/sentence/all-MiniLM-L6-v2
ENV TRANSFORMERS_OFFLINE=1
ENV HUGGINGFACE_HUB_OFFLINE=1
# --- end model block ---

# Copy source
COPY . .

# App Service listens on WEBSITES_PORT (set via app settings). Expose for local runs.
EXPOSE 8000

# Start the app
CMD ["python", "-m", "webapp.Smart_Elections_Parser_Webapp"]