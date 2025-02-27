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
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:                                                                                                                                      
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