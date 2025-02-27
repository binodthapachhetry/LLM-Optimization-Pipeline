"""                                                                                                                                                                                   
Quantization module for LLM optimization.                                                                                                                                             
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import logging                                                                                                                                                                        
from typing import Dict, Any, Optional, Tuple                                                                                                                                         
                                                                                                                                                                                    
import torch                                                                                                                                                                          
import transformers                                                                                                                                                                   
from transformers import AutoModelForCausalLM, AutoTokenizer                                                                                                                          
import bitsandbytes as bnb                                                                                                                                                            
from bitsandbytes.nn import Linear8bitLt, Linear4bit                                                                                                                                  
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage                                                                                                                                      
from llm_optimizer.utils.model import load_model_and_tokenizer                                                                                                                        
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
class QuantizationStage(OptimizationStage):                                                                                                                                           
    """                                                                                                                                                                               
    Quantization stage for LLM optimization.                                                                                                                                          
                                                                                                                                                                                    
    Supports various quantization methods:                                                                                                                                            
    - 8-bit quantization (using bitsandbytes)                                                                                                                                         
    - 4-bit quantization (using bitsandbytes)                                                                                                                                         
    - Dynamic quantization (using PyTorch)                                                                                                                                            
    - Static quantization (using PyTorch)                                                                                                                                             
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                     
        """                                                                                                                                                                           
        Run the quantization stage.                                                                                                                                                   
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_state: Current state of the model                                                                                                                                   
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Updated model state and metrics                                                                                                                                           
        """                                                                                                                                                                           
        self.validate_input(model_state)                                                                                                                                              
                                                                                                                                                                                    
        # Extract configuration                                                                                                                                                       
        model_path = model_state["model_path"]                                                                                                                                        
        method = self.config.get("method", "int8")                                                                                                                                    
        output_dir = os.path.join(                                                                                                                                                    
            self.config.get("output_dir", "./outputs"),                                                                                                                               
            f"quantized_{method}"                                                                                                                                                     
        )                                                                                                                                                                             
                                                                                                                                                                                    
        logger.info(f"Quantizing model {model_path} using {method}")                                                                                                                  
                                                                                                                                                                                    
        # Load model and tokenizer                                                                                                                                                    
        model, tokenizer = load_model_and_tokenizer(model_path)                                                                                                                       
                                                                                                                                                                                    
        # Apply quantization method                                                                                                                                                   
        if method == "int8":                                                                                                                                                          
            quantized_model = self._quantize_int8(model)                                                                                                                              
        elif method == "int4":                                                                                                                                                        
            quantized_model = self._quantize_int4(model)                                                                                                                              
        elif method == "dynamic":                                                                                                                                                     
            quantized_model = self._quantize_dynamic(model)                                                                                                                           
        elif method == "static":                                                                                                                                                      
            quantized_model = self._quantize_static(model)                                                                                                                            
        else:                                                                                                                                                                         
            raise ValueError(f"Unsupported quantization method: {method}")                                                                                                            
                                                                                                                                                                                    
        # Save the quantized model                                                                                                                                                    
        os.makedirs(output_dir, exist_ok=True)                                                                                                                                        
        quantized_model.save_pretrained(output_dir)                                                                                                                                   
        tokenizer.save_pretrained(output_dir)                                                                                                                                         
                                                                                                                                                                                    
        # Save quantization configuration                                                                                                                                             
        with open(os.path.join(output_dir, "quantization_config.txt"), "w") as f:                                                                                                     
            f.write(f"Quantization method: {method}\n")                                                                                                                               
            f.write(f"Original model: {model_path}\n")                                                                                                                                
                                                                                                                                                                                    
        logger.info(f"Quantized model saved to {output_dir}")                                                                                                                         
                                                                                                                                                                                    
        # Calculate model size reduction                                                                                                                                              
        original_size = self._get_model_size(model)                                                                                                                                   
        quantized_size = self._get_model_size(quantized_model)                                                                                                                        
        size_reduction = (original_size - quantized_size) / original_size * 100                                                                                                       
                                                                                                                                                                                    
        # Return updated model state and metrics                                                                                                                                      
        return {                                                                                                                                                                      
            "model_state": {                                                                                                                                                          
                "model_path": output_dir,                                                                                                                                             
                "is_pretrained": True,                                                                                                                                                
                "quantization_method": method,                                                                                                                                        
            },                                                                                                                                                                        
            "metrics": {                                                                                                                                                              
                "original_size_mb": original_size,                                                                                                                                    
                "quantized_size_mb": quantized_size,                                                                                                                                  
                "size_reduction_percent": size_reduction,                                                                                                                             
            },                                                                                                                                                                        
            "artifacts": {                                                                                                                                                            
                "model_path": output_dir,                                                                                                                                             
                "tokenizer_path": output_dir,                                                                                                                                         
                "config_path": os.path.join(output_dir, "quantization_config.txt"),                                                                                                   
            }                                                                                                                                                                         
        }                                                                                                                                                                             
                                                                                                                                                                                    
    def _quantize_int8(self, model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:                                                                                    
        """                                                                                                                                                                           
        Quantize model to 8-bit precision using bitsandbytes.                                                                                                                         
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to quantize                                                                                                                                              
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Quantized model                                                                                                                                                           
        """                                                                                                                                                                           
        # For a real implementation, we would load the model directly with load_in_8bit=True                                                                                          
        # Here we're simulating the process for an already loaded model                                                                                                               
                                                                                                                                                                                    
        # Create a new config for the 8-bit model                                                                                                                                     
        model_config = model.config                                                                                                                                                   
        model_config.quantization_config = {                                                                                                                                          
            "load_in_8bit": True,                                                                                                                                                     
            "llm_int8_threshold": 6.0,                                                                                                                                                
        }                                                                                                                                                                             
                                                                                                                                                                                    
        # Save and reload the model with 8-bit quantization                                                                                                                           
        temp_dir = os.path.join(self.config.get("output_dir", "./outputs"), "temp_model")                                                                                             
        os.makedirs(temp_dir, exist_ok=True)                                                                                                                                          
                                                                                                                                                                                    
        model.save_pretrained(temp_dir)                                                                                                                                               
                                                                                                                                                                                    
        # Load the model with 8-bit quantization                                                                                                                                      
        quantized_model = AutoModelForCausalLM.from_pretrained(                                                                                                                       
            temp_dir,                                                                                                                                                                 
            load_in_8bit=True,                                                                                                                                                        
            device_map="auto",                                                                                                                                                        
        )                                                                                                                                                                             
                                                                                                                                                                                    
        return quantized_model                                                                                                                                                        
                                                                                                                                                                                    
    def _quantize_int4(self, model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:                                                                                    
        """                                                                                                                                                                           
        Quantize model to 4-bit precision using bitsandbytes.                                                                                                                         
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to quantize                                                                                                                                              
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Quantized model                                                                                                                                                           
        """                                                                                                                                                                           
        # For a real implementation, we would load the model directly with load_in_4bit=True                                                                                          
        # Here we're simulating the process for an already loaded model                                                                                                               
                                                                                                                                                                                    
        # Create a new config for the 4-bit model                                                                                                                                     
        model_config = model.config                                                                                                                                                   
        model_config.quantization_config = {                                                                                                                                          
            "load_in_4bit": True,                                                                                                                                                     
            "bnb_4bit_compute_dtype": "float16",                                                                                                                                      
            "bnb_4bit_quant_type": "nf4",                                                                                                                                             
        }                                                                                                                                                                             
                                                                                                                                                                                    
        # Save and reload the model with 4-bit quantization                                                                                                                           
        temp_dir = os.path.join(self.config.get("output_dir", "./outputs"), "temp_model")                                                                                             
        os.makedirs(temp_dir, exist_ok=True)                                                                                                                                          
                                                                                                                                                                                    
        model.save_pretrained(temp_dir)                                                                                                                                               
                                                                                                                                                                                    
        # Load the model with 4-bit quantization                                                                                                                                      
        quantized_model = AutoModelForCausalLM.from_pretrained(                                                                                                                       
            temp_dir,                                                                                                                                                                 
            load_in_4bit=True,                                                                                                                                                        
            bnb_4bit_compute_dtype=torch.float16,                                                                                                                                     
            bnb_4bit_quant_type="nf4",                                                                                                                                                
            device_map="auto",                                                                                                                                                        
        )                                                                                                                                                                             
                                                                                                                                                                                    
        return quantized_model                                                                                                                                                        
                                                                                                                                                                                    
    def _quantize_dynamic(self, model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:                                                                                 
        """                                                                                                                                                                           
        Apply dynamic quantization using PyTorch.                                                                                                                                     
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to quantize                                                                                                                                              
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Quantized model                                                                                                                                                           
        """                                                                                                                                                                           
        # PyTorch dynamic quantization                                                                                                                                                
        # Note: This is a simplified version and may not work for all models                                                                                                          
                                                                                                                                                                                    
        # Move model to CPU for quantization                                                                                                                                          
        model = model.cpu()                                                                                                                                                           
                                                                                                                                                                                    
        # Apply dynamic quantization                                                                                                                                                  
        quantized_model = torch.quantization.quantize_dynamic(                                                                                                                        
            model,                                                                                                                                                                    
            {torch.nn.Linear},                                                                                                                                                        
            dtype=torch.qint8                                                                                                                                                         
        )                                                                                                                                                                             
                                                                                                                                                                                    
        return quantized_model                                                                                                                                                        
                                                                                                                                                                                    
    def _quantize_static(self, model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:                                                                                  
        """                                                                                                                                                                           
        Apply static quantization using PyTorch.                                                                                                                                      
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to quantize                                                                                                                                              
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Quantized model                                                                                                                                                           
        """                                                                                                                                                                           
        # Static quantization requires calibration data and model preparation                                                                                                         
        # This is a simplified placeholder implementation                                                                                                                             
                                                                                                                                                                                    
        # Move model to CPU for quantization                                                                                                                                          
        model = model.cpu()                                                                                                                                                           
                                                                                                                                                                                    
        # In a real implementation, we would:                                                                                                                                         
        # 1. Prepare the model for static quantization (add observers)                                                                                                                
        # 2. Calibrate with representative data                                                                                                                                       
        # 3. Convert to quantized model                                                                                                                                               
                                                                                                                                                                                    
        # For now, we'll just return the original model with a note                                                                                                                   
        logger.warning("Static quantization requires calibration data and model preparation. Returning original model.")                                                              
                                                                                                                                                                                    
        return model                                                                                                                                                                  
                                                                                                                                                                                    
    def _get_model_size(self, model: transformers.PreTrainedModel) -> float:                                                                                                          
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