# Use NVIDIA CUDA base image compatible with TensorFlow 2.15
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && \
    # Link python3.11 to python and python3
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .
COPY models/ ./models/

# Expose the port the app runs on
EXPOSE 5003

# Command to run the application using Uvicorn
# This is the production-ready way to start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5003"]
