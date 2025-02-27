"""                                                                                                                                                                                   
ONNX conversion module for LLM optimization.                                                                                                                                          
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import logging                                                                                                                                                                        
from typing import Dict, Any, Optional, Tuple                                                                                                                                         
                                                                                                                                                                                    
import torch                                                                                                                                                                          
from transformers import AutoModelForCausalLM, AutoTokenizer                                                                                                                          
from optimum.onnxruntime import ORTModelForCausalLM                                                                                                                                   
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage                                                                                                                                      
from llm_optimizer.utils.model import load_model_and_tokenizer                                                                                                                        
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
class ONNXConversionStage(OptimizationStage):                                                                                                                                         
    """                                                                                                                                                                               
    ONNX conversion stage for LLM optimization.                                                                                                                                       
                                                                                                                                                                                    
    Converts models to ONNX format for deployment and inference optimization.                                                                                                         
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                     
        """                                                                                                                                                                           
        Run the ONNX conversion stage.                                                                                                                                                
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_state: Current state of the model                                                                                                                                   
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Updated model state and metrics                                                                                                                                           
        """                                                                                                                                                                           
        self.validate_input(model_state)                                                                                                                                              
                                                                                                                                                                                    
        # Extract configuration                                                                                                                                                       
        model_path = model_state["model_path"]                                                                                                                                        
        output_dir = os.path.join(                                                                                                                                                    
            self.config.get("output_dir", "./outputs"),                                                                                                                               
            "onnx_model"                                                                                                                                                              
        )                                                                                                                                                                             
                                                                                                                                                                                    
        logger.info(f"Converting model {model_path} to ONNX format")                                                                                                                  
                                                                                                                                                                                    
        # Load model and tokenizer                                                                                                                                                    
        model, tokenizer = load_model_and_tokenizer(model_path)                                                                                                                       
                                                                                                                                                                                    
        # Convert to ONNX                                                                                                                                                             
        onnx_model = self._convert_to_onnx(model, tokenizer, output_dir)                                                                                                              
                                                                                                                                                                                    
        logger.info(f"ONNX model saved to {output_dir}")                                                                                                                              
                                                                                                                                                                                    
        # Return updated model state and metrics                                                                                                                                      
        return {                                                                                                                                                                      
            "model_state": {                                                                                                                                                          
                "model_path": output_dir,                                                                                                                                             
                "is_pretrained": True,                                                                                                                                                
                "format": "onnx",                                                                                                                                                     
            },                                                                                                                                                                        
            "metrics": {                                                                                                                                                              
                "conversion_success": True,                                                                                                                                           
            },                                                                                                                                                                        
            "artifacts": {                                                                                                                                                            
                "model_path": output_dir,                                                                                                                                             
            }                                                                                                                                                                         
        }                                                                                                                                                                             
                                                                                                                                                                                    
    def _convert_to_onnx(self, model, tokenizer, output_dir):                                                                                                                         
        """                                                                                                                                                                           
        Convert a model to ONNX format.                                                                                                                                               
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to convert                                                                                                                                               
            tokenizer: The tokenizer                                                                                                                                                  
            output_dir: Directory to save the ONNX model                                                                                                                              
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Path to the converted model                                                                                                                                               
        """                                                                                                                                                                           
        try:                                                                                                                                                                          
            # Method 1: Using Optimum (preferred)                                                                                                                                     
            return self._convert_with_optimum(model, tokenizer, output_dir)                                                                                                           
        except Exception as e:                                                                                                                                                        
            logger.warning(f"Failed to convert with Optimum: {e}")                                                                                                                    
                                                                                                                                                                                    
            # Method 2: Using PyTorch's built-in ONNX export                                                                                                                          
            return self._convert_with_torch(model, tokenizer, output_dir)                                                                                                             
                                                                                                                                                                                    
    def _convert_with_optimum(self, model, tokenizer, output_dir):                                                                                                                    
        """Convert using Optimum library."""                                                                                                                                          
        from optimum.onnxruntime import ORTModelForCausalLM                                                                                                                           
                                                                                                                                                                                    
        # Create output directory                                                                                                                                                     
        os.makedirs(output_dir, exist_ok=True)                                                                                                                                        
                                                                                                                                                                                    
        # Save model and tokenizer temporarily if needed                                                                                                                              
        if isinstance(model_path := getattr(model, "name_or_path", None), str):                                                                                                       
            # Model is already a path or has a path attribute                                                                                                                         
            pass                                                                                                                                                                      
        else:                                                                                                                                                                         
            # Save model temporarily                                                                                                                                                  
            temp_dir = os.path.join(output_dir, "temp_model")                                                                                                                         
            os.makedirs(temp_dir, exist_ok=True)                                                                                                                                      
            model.save_pretrained(temp_dir)                                                                                                                                           
            tokenizer.save_pretrained(temp_dir)                                                                                                                                       
            model_path = temp_dir                                                                                                                                                     
                                                                                                                                                                                    
        # Convert to ONNX                                                                                                                                                             
        onnx_model = ORTModelForCausalLM.from_pretrained(                                                                                                                             
            model_path,                                                                                                                                                               
            export=True,                                                                                                                                                              
            provider="CPUExecutionProvider"                                                                                                                                           
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Save ONNX model                                                                                                                                                             
        onnx_model.save_pretrained(output_dir)                                                                                                                                        
        tokenizer.save_pretrained(output_dir)                                                                                                                                         
                                                                                                                                                                                    
        return output_dir                                                                                                                                                             
                                                                                                                                                                                    
    def _convert_with_torch(self, model, tokenizer, output_dir):                                                                                                                      
        """Convert using PyTorch's ONNX export."""                                                                                                                                    
        import torch.onnx                                                                                                                                                             
                                                                                                                                                                                    
        # Create output directory                                                                                                                                                     
        os.makedirs(output_dir, exist_ok=True)                                                                                                                                        
                                                                                                                                                                                    
        # Prepare dummy input                                                                                                                                                         
        batch_size = 1                                                                                                                                                                
        sequence_length = 128                                                                                                                                                         
        dummy_input = torch.randint(                                                                                                                                                  
            100, 1000,                                                                                                                                                                
            (batch_size, sequence_length),                                                                                                                                            
            device=model.device                                                                                                                                                       
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Set model to evaluation mode                                                                                                                                                
        model.eval()                                                                                                                                                                  
                                                                                                                                                                                    
        # Export model to ONNX                                                                                                                                                        
        torch.onnx.export(                                                                                                                                                            
            model,                                                                                                                                                                    
            dummy_input,                                                                                                                                                              
            os.path.join(output_dir, "model.onnx"),                                                                                                                                   
            export_params=True,                                                                                                                                                       
            opset_version=13,                                                                                                                                                         
            do_constant_folding=True,                                                                                                                                                 
            input_names=["input_ids"],                                                                                                                                                
            output_names=["logits"],                                                                                                                                                  
            dynamic_axes={                                                                                                                                                            
                "input_ids": {0: "batch_size", 1: "sequence_length"},                                                                                                                 
                "logits": {0: "batch_size", 1: "sequence_length"},                                                                                                                    
            },                                                                                                                                                                        
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Save tokenizer                                                                                                                                                              
        tokenizer.save_pretrained(output_dir)                                                                                                                                         
                                                                                                                                                                                    
        # Save config                                                                                                                                                                 
        if hasattr(model, "config"):                                                                                                                                                  
            model.config.save_pretrained(output_dir)                                                                                                                                  
                                                                                                                                                                                    
        return output_dir 