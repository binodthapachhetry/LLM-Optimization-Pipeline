"""                                                                                                                                                                                   
Model utility functions for LLM optimization.                                                                                                                                         
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import logging                                                                                                                                                                        
from typing import Dict, Any, Optional, Tuple, Union                                                                                                                                  
                                                                                                                                                                                    
import torch                                                                                                                                                                          
from transformers import AutoModelForCausalLM, AutoTokenizer                                                                                                                          
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
def load_model_and_tokenizer(                                                                                                                                                         
    model_path: str,                                                                                                                                                                  
    load_in_8bit: bool = False,                                                                                                                                                       
    load_in_4bit: bool = False,                                                                                                                                                       
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,                                                                                                              
    torch_dtype: Optional[torch.dtype] = None,                                                                                                                                        
) -> Tuple[Any, Any]:                                                                                                                                                      
    """                                                                                                                                                                               
    Load a model and tokenizer from a path or model name.                                                                                                                             
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        model_path: Path to model or model name                                                                                                                                       
        load_in_8bit: Whether to load in 8-bit precision                                                                                                                              
        load_in_4bit: Whether to load in 4-bit precision                                                                                                                              
        device_map: Device map for model parallelism                                                                                                                                  
        torch_dtype: Data type for model weights                                                                                                                                      
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Tuple of (model, tokenizer)                                                                                                                                                   
    """                                                                                                                                                                               
    logger.info(f"Loading model from {model_path}")                                                                                                                                   
    
    # Check if this is a GGUF model
    if model_path.endswith(".gguf"):
        return load_gguf_model(model_path, device_map)
                                                                                                                                                                                    
    # Set default device map if not provided                                                                                                                                          
    if device_map is None:                                                                                                                                                            
        if torch.cuda.is_available():                                                                                                                                                 
            device_map = "auto"                                                                                                                                                       
        else:                                                                                                                                                                         
            device_map = "cpu"                                                                                                                                                        
                                                                                                                                                                                    
    # Set default torch dtype if not provided                                                                                                                                         
    if torch_dtype is None:                                                                                                                                                           
        if torch.cuda.is_available():                                                                                                                                                 
            torch_dtype = torch.float16                                                                                                                                               
        else:                                                                                                                                                                         
            torch_dtype = torch.float32                                                                                                                                               
                                                                                                                                                                                    
    # Load tokenizer                                                                                                                                                                  
    tokenizer = AutoTokenizer.from_pretrained(                                                                                                                                        
        model_path,                                                                                                                                                                   
        trust_remote_code=True,                                                                                                                                                       
    )                                                                                                                                                                                 
                                                                                                                                                                                    
    # Ensure padding token is set                                                                                                                                                     
    if tokenizer.pad_token is None:                                                                                                                                                   
        tokenizer.pad_token = tokenizer.eos_token                                                                                                                                     
                                                                                                                                                                                    
    # Load model with appropriate quantization                                                                                                                                        
    if load_in_8bit:                                                                                                                                                                  
        model = AutoModelForCausalLM.from_pretrained(                                                                                                                                 
            model_path,                                                                                                                                                               
            load_in_8bit=True,                                                                                                                                                        
            device_map=device_map,                                                                                                                                                    
            trust_remote_code=True,                                                                                                                                                   
        )                                                                                                                                                                             
    elif load_in_4bit:                                                                                                                                                                
        model = AutoModelForCausalLM.from_pretrained(                                                                                                                                 
            model_path,                                                                                                                                                               
            load_in_4bit=True,                                                                                                                                                        
            device_map=device_map,                                                                                                                                                    
            trust_remote_code=True,                                                                                                                                                   
        )                                                                                                                                                                             
    else:                                                                                                                                                                             
        model = AutoModelForCausalLM.from_pretrained(                                                                                                                                 
            model_path,                                                                                                                                                               
            device_map=device_map,                                                                                                                                                    
            torch_dtype=torch_dtype,                                                                                                                                                  
            trust_remote_code=True,                                                                                                                                                   
        )                                                                                                                                                                             
                                                                                                                                                                                    
    logger.info(f"Model loaded successfully")                                                                                                                                         
                                                                                                                                                                                    
    return model, tokenizer                                                                                                                                                           


def load_gguf_model(model_path: str, device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None) -> Tuple[Any, Any]:
    """
    Load a GGUF model using llama-cpp-python.
    
    Args:
        model_path: Path to the GGUF model file
        device_map: Device map (only 'cpu' or 'cuda' supported)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is required to load GGUF models. "
            "Install it with: pip install llama-cpp-python"
        )
    
    # Determine if we should use GPU
    use_gpu = False
    n_gpu_layers = 0
    
    if device_map is not None:
        if isinstance(device_map, str):
            use_gpu = device_map == "cuda" or device_map == "auto" and torch.cuda.is_available()
        elif isinstance(device_map, dict):
            # If any layer is on GPU, enable GPU
            use_gpu = any(v == "cuda" or isinstance(v, int) for v in device_map.values())
    
    if use_gpu:
        n_gpu_layers = -1  # Use all layers on GPU
    
    # Load the model
    logger.info(f"Loading GGUF model with n_gpu_layers={n_gpu_layers}")
    llama_model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048,  # Context window size
    )
    
    # Create wrapper classes to make the GGUF model compatible with our benchmarking interface
    model = GGUFModelWrapper(llama_model)
    tokenizer = GGUFTokenizerWrapper(llama_model)
    
    logger.info(f"GGUF model loaded successfully")
    
    return model, tokenizer


class GGUFModelWrapper:
    """Wrapper for GGUF models to make them compatible with the benchmarking interface."""
    
    def __init__(self, llama_model):
        self.model = llama_model
        self.device = torch.device("cuda" if getattr(llama_model, "n_gpu_layers", 0) > 0 else "cpu")
    
    def __call__(self, input_ids, **kwargs):
        """
        Forward pass that mimics the HF interface.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Object with a 'logits' attribute
        """
        # Convert to list of token IDs if it's a tensor
        if hasattr(input_ids, "cpu"):
            input_ids = input_ids.cpu().numpy().tolist()
        
        if isinstance(input_ids, list) and isinstance(input_ids[0], list):
            # Handle batched inputs - we'll process just the first one for simplicity
            # In a real implementation, you'd want to process all batch items
            input_ids = input_ids[0]
        
        # Run the model (just evaluate without generating)
        self.model.eval(input_ids)
        
        # Create a simple object with logits attribute to match HF interface
        class SimpleOutput:
            def __init__(self, logits):
                self.logits = logits
        
        # Create fake logits tensor of the right shape
        # This is a simplification - real logits would be the actual model outputs
        vocab_size = self.model.n_vocab()
        seq_len = len(input_ids)
        logits = torch.zeros((1, seq_len, vocab_size), device=self.device)
        
        return SimpleOutput(logits)
    
    def eval(self):
        """Set the model to evaluation mode."""
        # GGUF models are always in eval mode
        return self
    
    def parameters(self):
        """Return an empty iterator for parameters."""
        # This is needed for the memory calculation
        # GGUF models don't expose parameters in the same way
        return []
    
    def buffers(self):
        """Return an empty iterator for buffers."""
        # This is needed for the memory calculation
        return []
    
    @property
    def name_or_path(self):
        """Return the model path."""
        return getattr(self.model, "model_path", "gguf-model")


class GGUFTokenizerWrapper:
    """Wrapper for GGUF model tokenizer to make it compatible with the HF interface."""
    
    def __init__(self, llama_model):
        self.model = llama_model
    
    def __call__(self, text, **kwargs):
        """Tokenize text."""
        if isinstance(text, list):
            # Handle batch
            return [self.model.tokenize(t) for t in text]
        return self.model.tokenize(text)
    
    def decode(self, token_ids, **kwargs):
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy().tolist()
        
        # GGUF models typically don't have a batch decode method
        # so we'll implement a simple version
        if isinstance(token_ids[0], list):
            return [self.model.detokenize(ids) for ids in token_ids]
        return self.model.detokenize(token_ids)
    
    def save_pretrained(self, save_directory):
        """Mock save_pretrained method."""
        # We don't actually save anything, just create a dummy file
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            f.write('{"type": "gguf_wrapper"}')
                                                                                                                                                                                    
                                                                                                                                                                                    
def get_model_size(model: torch.nn.Module) -> float:                                                                                                                                  
    """                                                                                                                                                                               
    Calculate the size of a model in MB.                                                                                                                                              
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        model: The model to measure                                                                                                                                                   
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Size in MB                                                                                                                                                                    
    """                                                                                                                                                                               
    param_size = 0                                                                                                                                                                    
    for param in model.parameters():                                                                                                                                                  
        param_size += param.nelement() * param.element_size()                                                                                                                         
                                                                                                                                                                                    
    buffer_size = 0                                                                                                                                                                   
    for buffer in model.buffers():                                                                                                                                                    
        buffer_size += buffer.nelement() * buffer.element_size()                                                                                                                      
                                                                                                                                                                                    
    size_mb = (param_size + buffer_size) / 1024 / 1024                                                                                                                                
                                                                                                                                                                                    
    return size_mb                                                                                                                                                                    
                                                                                                                                                                                    
                                                                                                                                                                                    
def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:                                                                                                                         
    """                                                                                                                                                                               
    Get information about a model.                                                                                                                                                    
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        model: The model to analyze                                                                                                                                                   
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Dictionary of model information                                                                                                                                               
    """                                                                                                                                                                               
    # Count parameters                                                                                                                                                                
    total_params = sum(p.numel() for p in model.parameters())                                                                                                                         
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)                                                                                                  
                                                                                                                                                                                    
    # Get model size                                                                                                                                                                  
    model_size = get_model_size(model)                                                                                                                                                
                                                                                                                                                                                    
    # Get model device                                                                                                                                                                
    device = next(model.parameters()).device                                                                                                                                          
                                                                                                                                                                                    
    # Get model dtype                                                                                                                                                                 
    dtype = next(model.parameters()).dtype                                                                                                                                            
                                                                                                                                                                                    
    return {                                                                                                                                                                          
        "total_parameters": total_params,                                                                                                                                             
        "trainable_parameters": trainable_params,                                                                                                                                     
        "model_size_mb": model_size,                                                                                                                                                  
        "device": str(device),                                                                                                                                                        
        "dtype": str(dtype),                                                                                                                                                          
    }                                                                                                                                                                                 
                                                                                                                                                                                    
                                                                                                                                                                                    
def save_model_and_tokenizer(                                                                                                                                                         
    model: torch.nn.Module,                                                                                                                                                           
    tokenizer: AutoTokenizer,                                                                                                                                                         
    output_dir: str,                                                                                                                                                                  
    save_format: str = "pytorch",                                                                                                                                                     
) -> str:                                                                                                                                                                             
    """                                                                                                                                                                               
    Save a model and tokenizer.                                                                                                                                                       
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        model: The model to save                                                                                                                                                      
        tokenizer: The tokenizer to save                                                                                                                                              
        output_dir: Directory to save to                                                                                                                                              
        save_format: Format to save in (pytorch, safetensors)                                                                                                                         
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Path to saved model                                                                                                                                                           
    """                                                                                                                                                                               
    os.makedirs(output_dir, exist_ok=True)                                                                                                                                            
                                                                                                                                                                                    
    # Save model                                                                                                                                                                      
    if save_format == "safetensors":                                                                                                                                                  
        from safetensors.torch import save_file                                                                                                                                       
                                                                                                                                                                                    
        # Save model weights as safetensors                                                                                                                                           
        state_dict = model.state_dict()                                                                                                                                               
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))                                                                                                          
                                                                                                                                                                                    
        # Save model config                                                                                                                                                           
        if hasattr(model, "config"):                                                                                                                                                  
            model.config.save_pretrained(output_dir)                                                                                                                                  
    else:                                                                                                                                                                             
        # Save using Hugging Face's save_pretrained                                                                                                                                   
        model.save_pretrained(output_dir)                                                                                                                                             
                                                                                                                                                                                    
    # Save tokenizer                                                                                                                                                                  
    tokenizer.save_pretrained(output_dir)                                                                                                                                             
                                                                                                                                                                                    
    logger.info(f"Model and tokenizer saved to {output_dir}")                                                                                                                         
                                                                                                                                                                                    
    return output_dir 
