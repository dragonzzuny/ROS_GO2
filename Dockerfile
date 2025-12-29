# Multi-stage Dockerfile for RL Dispatch MVP
# Provides reproducible environment for training and deployment

# Base image with Python and CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS base

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install Python package
RUN pip install --no-cache-dir -e .

# Development stage with additional tools
FROM base AS dev

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy \
    ipython \
    jupyter \
    matplotlib \
    seaborn

# Production stage (minimal)
FROM base AS prod

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Set entrypoint
ENTRYPOINT ["python"]
CMD ["scripts/train.py"]

# Expose TensorBoard port
EXPOSE 6006
