# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in stages to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.14.0 numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 && \
    pip install --no-cache-dir fastapi==0.100.1 uvicorn[standard]==0.23.2 pydantic==2.4.2 python-multipart==0.0.6 && \
    pip install --no-cache-dir keras-tcn==3.5.0 python-json-logger==2.0.7 requests==2.31.0 && \
    pip install --no-cache-dir psutil==5.9.6 GPUtil==1.4.0

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5003/v3/health || exit 1

# Run the application
CMD ["python", "main.py"]