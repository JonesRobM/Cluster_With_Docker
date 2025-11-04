# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY evaluate.py .

# Create output directories
RUN mkdir -p outputs/plots outputs/metrics

# Set Python path
ENV PYTHONPATH=/app

# Run evaluation script by default
CMD ["python", "evaluate.py"]
