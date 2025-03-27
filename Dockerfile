# Base image with PyTorch and CUDA 12.8.0
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with proper permissions
RUN mkdir -p /root/.cache/huggingface /root/.cache/llm_optimizer /root/.cache/torch

# Set environment variables for GPU and CUDA
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1
# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python


# Create .dockerignore if it doesn't exist
COPY .dockerignore* ./

# Copy only necessary files (excluding those in .dockerignore)
COPY . .

# Install the package in development mode
RUN pip install -e .

# Verify CUDA is available
RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda)"

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Default command
ENTRYPOINT ["llm-optimizer"]
CMD ["--help"]
