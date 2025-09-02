# Base image for Azure App Service (Python 3.12)
FROM mcr.microsoft.com/azure-app-service/python:3.12

# System deps for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /home/site/wwwroot

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir playwright && \
    python -m playwright install --with-deps

# Copy source
COPY . .

# Enable OCR in container
ENV ENABLE_OCR=True
ENV PYTHONUNBUFFERED=1

# Expose default port used by Azure images (App Service sets PORT)
EXPOSE 8000

# Start the app
CMD ["python", "-m", "webapp.Smart_Elections_Parser_Webapp"]