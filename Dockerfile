# Use the official Python slim image for a smaller footprint
FROM python:3.11-slim

# Install system dependencies, specifically poppler-utils for pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

# Copy the rest of the application
COPY . .

# Expose a default port (Render overrides this with the $PORT environment variable)
EXPOSE 8000

# The CMD to start the backend (downloads the model, then starts FastAPI)
CMD ["sh", "-c", "python backend/download_model.py && python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
