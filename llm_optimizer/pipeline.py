"""                                                                                                                                                                                   
Core pipeline implementation for LLM optimization.                                                                                                                                    
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import time                                                                                                                                                                           
import logging                                                                                                                                                                        
from typing import Dict, Any, List, Optional, Type                                                                                                                                    
                                                                                                                                                                                    
from omegaconf import DictConfig                                                                                                                                                      
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage                                                                                                                                      
from llm_optimizer.fine_tuning import FineTuningStage                                                                                                                                 
from llm_optimizer.quantization import QuantizationStage                                                                                                                              
from llm_optimizer.pruning import PruningStage                                                                                                                                        
from llm_optimizer.distillation import DistillationStage                                                                                                                              
from llm_optimizer.evaluation import EvaluationStage                                                                                                                                  
from llm_optimizer.onnx_conversion import ONNXConversionStage                                                                                                                         
from llm_optimizer.benchmarking import BenchmarkingStage
from llm_optimizer.prompt_optimization import PromptOptimizationStage                                                                                                                        
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
class OptimizationPipeline:                                                                                                                                                           
    """                                                                                                                                                                               
    A modular pipeline for LLM optimization.                                                                                                                                          
                                                                                                                                                                                    
    This class orchestrates the execution of various optimization stages                                                                                                              
    such as fine-tuning, quantization, pruning, distillation, and evaluation.                                                                                                         
    """                                                                                                                                                                               
                                                                                                                                                                                    
    # Registry of available stages                                                                                                                                                    
    STAGE_REGISTRY: Dict[str, Type[OptimizationStage]] = {                                                                                                                            
        "fine_tuning": FineTuningStage,                                                                                                                                               
        "quantization": QuantizationStage,                                                                                                                                            
        "pruning": PruningStage,                                                                                                                                                      
        "distillation": DistillationStage,                                                                                                                                            
        "evaluation": EvaluationStage,                                                                                                                                                
        "onnx_conversion": ONNXConversionStage,                                                                                                                                       
        "benchmarking": BenchmarkingStage,  
        "prompt_optimization": PromptOptimizationStage,                                                                                                                                          
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    def __init__(self, config: DictConfig):                                                                                                                                           
        """                                                                                                                                                                           
        Initialize the optimization pipeline.                                                                                                                                         
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            config: Configuration for the pipeline                                                                                                                                    
        """                                                                                                                                                                           
        self.config = config                                                                                                                                                          
        self.stages: List[OptimizationStage] = []                                                                                                                                     
        self._initialize_stages()                                                                                                                                                     
                                                                                                                                                                                    
    def _initialize_stages(self) -> None:                                                                                                                                             
        """Initialize the pipeline stages based on configuration."""                                                                                                                  
        stage_names = self.config.pipeline.stages                                                                                                                                     
                                                                                                                                                                                    
        for stage_name in stage_names:                                                                                                                                                
            if stage_name not in self.STAGE_REGISTRY:                                                                                                                                 
                raise ValueError(f"Unknown stage: {stage_name}. Available stages: {list(self.STAGE_REGISTRY.keys())}")                                                                
                                                                                                                                                                                    
            # Get stage class and create instance                                                                                                                                     
            stage_class = self.STAGE_REGISTRY[stage_name]                                                                                                                             
            stage_config = self.config.get(stage_name, {})                                                                                                                            
            stage = stage_class(stage_config)                                                                                                                                         
                                                                                                                                                                                    
            self.stages.append(stage)                                                                                                                                                 
                                                                                                                                                                                    
        logger.info(f"Initialized pipeline with {len(self.stages)} stages: {stage_names}")                                                                                            
                                                                                                                                                                                    
    def run(self) -> Dict[str, Any]:                                                                                                                                                  
        """                                                                                                                                                                           
        Run the optimization pipeline.                                                                                                                                                
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary containing results and metrics                                                                                                                                 
        """                                                                                                                                                                           
        start_time = time.time()                                                                                                                                                      
        results = {                                                                                                                                                                   
            "completed_stages": [],                                                                                                                                                   
            "metrics": {},                                                                                                                                                            
            "artifacts": {},                                                                                                                                                          
        }                                                                                                                                                                             
                                                                                                                                                                                    
        # Current model state that gets passed between stages                                                                                                                         
        model_state = {                                                                                                                                                               
            "model_path": self.config.model.name,                                                                                                                                     
            "is_pretrained": self.config.model.get("pretrained", True),                                                                                                               
        }                                                                                                                                                                             
                                                                                                                                                                                    
        # Run each stage in sequence                                                                                                                                                  
        for i, stage in enumerate(self.stages):                                                                                                                                       
            stage_name = self.config.pipeline.stages[i]                                                                                                                               
            logger.info(f"Running stage {i+1}/{len(self.stages)}: {stage_name}")                                                                                                      
                                                                                                                                                                                    
            try:                                                                                                                                                                      
                # Run the stage                                                                                                                                                       
                stage_output = stage.run(model_state)                                                                                                                                 
                                                                                                                                                                                    
                # Update model state for next stage                                                                                                                                   
                model_state.update(stage_output.get("model_state", {}))                                                                                                               
                                                                                                                                                                                    
                # Collect results                                                                                                                                                     
                results["completed_stages"].append(stage_name)                                                                                                                        
                results["metrics"][stage_name] = stage_output.get("metrics", {})                                                                                                      
                results["artifacts"][stage_name] = stage_output.get("artifacts", {})                                                                                                  
                                                                                                                                                                                    
                logger.info(f"Completed stage: {stage_name}")                                                                                                                         
                                                                                                                                                                                    
            except Exception as e:                                                                                                                                                    
                logger.error(f"Error in stage {stage_name}: {str(e)}", exc_info=True)                                                                                                 
                results["error"] = {                                                                                                                                                  
                    "stage": stage_name,                                                                                                                                              
                    "message": str(e),                                                                                                                                                
                }                                                                                                                                                                     
                break                                                                                                                                                                 
                                                                                                                                                                                    
        # Calculate total time                                                                                                                                                        
        end_time = time.time()                                                                                                                                                        
        results["total_time"] = end_time - start_time                                                                                                                                 
                                                                                                                                                                                    
        return results                                                                                                                                                                
                                                                                                                                                                                    
    def get_stage(self, stage_name: str) -> Optional[OptimizationStage]:                                                                                                              
        """                                                                                                                                                                           
        Get a stage by name.                                                                                                                                                          
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            stage_name: Name of the stage to get                                                                                                                                      
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            The stage instance or None if not found                                                                                                                                   
        """                                                                                                                                                                           
        stage_names = self.config.pipeline.stages                                                                                                                                     
                                                                                                                                                                                    
        for i, name in enumerate(stage_names):                                                                                                                                        
            if name == stage_name:                                                                                                                                                    
                return self.stages[i]                                                                                                                                                 
                                                                                                                                                                                    
        return None  