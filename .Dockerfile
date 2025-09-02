# Use a stable, public base
FROM python:3.12-slim-bookworm

# System deps for OCR and Playwright
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir playwright && \
    python -m playwright install --with-deps

# Copy source
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV ENABLE_OCR=True

# App Service listens on WEBSITES_PORT (set via app settings). Expose for local runs.
EXPOSE 8000

# Start the app
CMD ["python", "-m", "webapp.Smart_Elections_Parser_Webapp"]