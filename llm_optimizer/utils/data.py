"""
Data utility functions for LLM optimization.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


def get_bundled_dataset_path(dataset_name: str) -> str:
    """
    Get the path to a bundled dataset file.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'tiny_wikitext', 'tiny_lambada')
        
    Returns:
        Path to the dataset file
    """
    import os
    import pkg_resources
    
    if dataset_name == 'tiny_wikitext':
        return pkg_resources.resource_filename('llm_optimizer', 'data/evaluation/tiny_wikitext.txt')
    elif dataset_name == 'tiny_lambada':
        return pkg_resources.resource_filename('llm_optimizer', 'data/evaluation/tiny_lambada.json')
    else:
        raise ValueError(f"Unknown bundled dataset: {dataset_name}")


def load_bundled_dataset(dataset_name: str) -> Union[str, List[Dict[str, str]]]:
    """
    Load a bundled dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'tiny_wikitext', 'tiny_lambada')
        
    Returns:
        Dataset content (text for wikitext, list of dicts for lambada)
    """
    path = get_bundled_dataset_path(dataset_name)
    
    if dataset_name == 'tiny_wikitext':
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    elif dataset_name == 'tiny_lambada':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown bundled dataset: {dataset_name}")


def get_available_bundled_datasets() -> List[str]:
    """
    Get a list of available bundled datasets.
    
    Returns:
        List of dataset names
    """
    return ['tiny_wikitext', 'tiny_lambada']
