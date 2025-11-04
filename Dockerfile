# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml .
COPY MANIFEST.in .
COPY requirements.txt .

# Copy source code
COPY src/ ./src/
COPY evaluate.py .

# Install the package with all dependencies
RUN pip install --no-cache-dir .

# Create output directories
RUN mkdir -p outputs/plots outputs/metrics

# Run evaluation script by default
CMD ["python", "evaluate.py"]
