FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Pre-download the model during build to avoid runtime download
RUN python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
    print('Downloading Nemotron OCR v2 model...'); \
    AutoProcessor.from_pretrained('nvidia/nemotron-ocr-v2', trust_remote_code=True); \
    AutoModelForVision2Seq.from_pretrained('nvidia/nemotron-ocr-v2', trust_remote_code=True); \
    print('Model downloaded successfully!')"

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

# Made with Bob
