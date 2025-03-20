"""
Utilities for working with GGUF models.
"""

import os
import re
import logging
import tempfile
from typing import Tuple, Optional, Union
import requests
from huggingface_hub import hf_hub_download, HfApi
from llama_cpp import Llama
from llm_optimizer.utils.model import GGUFModelWrapper, GGUFTokenizerWrapper

from rich.console import Console  



console = Console() 

logger = logging.getLogger(__name__)

def download_gguf_model(model_path: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a GGUF model from various sources.
    
    Supports:
    - Local file paths
    - Hugging Face model repositories
    - Direct URLs
    - GitHub URLs
    
    Args:
        model_path: Path or URL to the GGUF model
        cache_dir: Directory to cache downloaded models
        
    Returns:
        Local path to the downloaded model
    """
    # If it's a local file that exists, return it directly
    if os.path.exists(model_path) and os.path.isfile(model_path):
        logger.info(f"Using local GGUF model: {model_path}")
        return model_path
        
    # If no cache directory specified, use the default cache
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "llm_optimizer", "gguf_models")
    
    os.makedirs(cache_dir, exist_ok=True)
    console.print(cache_dir)
    
    # Case 1: Hugging Face model repository
    if "huggingface.co" in model_path or not any(prefix in model_path for prefix in ["http://", "https://", "ftp://"]):
        return _download_from_huggingface(model_path, cache_dir)
    
    # Case 2: GitHub URL
    elif "github.com" in model_path:
        return _download_from_github(model_path, cache_dir)
    
    # Case 3: Direct URL
    elif model_path.startswith(("http://", "https://", "ftp://")):
        return _download_from_url(model_path, cache_dir)
    
    # If we get here, we don't know how to handle this path
    raise ValueError(f"Unsupported model path format: {model_path}")

def _download_from_huggingface(model_path: str, cache_dir: str) -> str:
    """Download a GGUF model from Hugging Face."""
    try:
        # Extract repo_id and filename
        if "huggingface.co" in model_path:
            # URL format: https://huggingface.co/repo_id/blob/main/filename.gguf
            match = re.search(r"huggingface\.co/([^/]+/[^/]+)(?:/blob/[^/]+)?/([^/]+\.gguf)", model_path)
            if match:
                repo_id, filename = match.groups()
            else:
                # Try another pattern
                match = re.search(r"huggingface\.co/([^/]+/[^/]+)", model_path)
                if match:
                    repo_id = match.group(1)
                    # Find a GGUF file in the repo
                    api = HfApi()
                    files = api.list_repo_files(repo_id)
                    gguf_files = [f for f in files if f.endswith('.gguf')]
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in repository {repo_id}")
                    filename = gguf_files[0]  # Use the first GGUF file
                else:
                    raise ValueError(f"Could not parse Hugging Face URL: {model_path}")
        else:
            # Direct repo_id format: repo_id or repo_id/filename.gguf
            parts = model_path.split('/')
            if len(parts) >= 2:
                if parts[-1].endswith('.gguf'):
                    repo_id = '/'.join(parts[:-1])
                    filename = parts[-1]
                else:
                    repo_id = model_path
                    # Find a GGUF file in the repo
                    api = HfApi()
                    files = api.list_repo_files(repo_id)
                    gguf_files = [f for f in files if f.endswith('.gguf')]
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in repository {repo_id}")
                    filename = gguf_files[0]  # Use the first GGUF file
            else:
                raise ValueError(f"Invalid Hugging Face repository format: {model_path}")
        
        logger.info(f"Downloading GGUF model from Hugging Face: {repo_id}/{filename}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        return local_path
        
    except Exception as e:
        logger.error(f"Error downloading from Hugging Face: {e}")
        raise

def _download_from_github(model_path: str, cache_dir: str) -> str:
    """Download a GGUF model from GitHub."""
    try:
        # Convert GitHub URL to raw content URL
        # From: https://github.com/user/repo/blob/branch/path/to/file.gguf
        # To:   https://raw.githubusercontent.com/user/repo/branch/path/to/file.gguf
        raw_url = model_path.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        
        return _download_from_url(raw_url, cache_dir)
        
    except Exception as e:
        logger.error(f"Error downloading from GitHub: {e}")
        raise

def _download_from_url(url: str, cache_dir: str) -> str:
    """Download a GGUF model from a direct URL."""
    try:
        # Extract filename from URL
        filename = os.path.basename(url.split("?")[0])  # Remove query parameters
        if not filename.endswith('.gguf'):
            filename = f"{filename}.gguf"
            
        # Create cache path
        cache_path = os.path.join(cache_dir, filename)
        
        # Check if file already exists in cache
        if os.path.exists(cache_path):
            logger.info(f"Using cached GGUF model: {cache_path}")
            return cache_path
            
        logger.info(f"Downloading GGUF model from URL: {url}")
        
        # Download with progress reporting
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            # Use a temporary file during download
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        # Log progress
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (100 * 1024 * 1024) == 0:  # Log every 100MB
                                logger.info(f"Downloaded {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB ({percent:.1f}%)")
                
                temp_path = temp_file.name
                
            # Move the temp file to the cache location
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            os.rename(temp_path, cache_path)
            
        logger.info(f"Downloaded GGUF model to: {cache_path}")
        return cache_path
        
    except Exception as e:
        logger.error(f"Error downloading from URL: {e}")
        raise

def load_gguf_model(model_path: str, n_ctx: int = 512, n_gpu_layers: int = -1) -> Tuple[GGUFModelWrapper, GGUFTokenizerWrapper]:
    """
    Load a GGUF model with automatic downloading.
    
    Args:
        model_path: Path or URL to the GGUF model
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        
    Returns:
        Tuple of (model_wrapper, tokenizer_wrapper)
    """
    # Download the model if needed
    console.print(model_path)
    local_path = download_gguf_model(model_path)

    console.print(local_path)
    
    # Load the model with llama-cpp-python
    llama_model = Llama(
        model_path=local_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers
    )

    console.print("out of llama")
    
    # Create wrappers for compatibility with the benchmarking interface
    model_wrapper = GGUFModelWrapper(llama_model)
    tokenizer_wrapper = GGUFTokenizerWrapper(llama_model)
    
    return model_wrapper, tokenizer_wrapper
