import pytest                                                                                                                                                                         
import os                                                                                                                                                                             
import torch                                                                                                                                                                          
from omegaconf import OmegaConf                                                                                                                                                       
                                                                                                                                                                                    
from llm_optimizer.pipeline import OptimizationPipeline                                                                                                                               
from llm_optimizer.base import OptimizationStage                                                                                                                                      
                                                                                                                                                                                    
                                                                                                                                                                                    
class MockStage(OptimizationStage):                                                                                                                                                   
    """Mock optimization stage for testing."""                                                                                                                                        
                                                                                                                                                                                    
    def run(self, model_state):                                                                                                                                                       
        """Run the mock stage."""                                                                                                                                                     
        self.validate_input(model_state)                                                                                                                                              
                                                                                                                                                                                    
        # Return unchanged model state and mock metrics                                                                                                                               
        return {                                                                                                                                                                      
            "model_state": model_state,                                                                                                                                               
            "metrics": {                                                                                                                                                              
                "mock_metric": 0.95,                                                                                                                                                  
            },                                                                                                                                                                        
            "artifacts": {                                                                                                                                                            
                "mock_artifact": "mock_path",                                                                                                                                         
            }                                                                                                                                                                         
        }                                                                                                                                                                             
                                                                                                                                                                                    
                                                                                                                                                                                    
def test_pipeline_initialization():                                                                                                                                                   
    """Test that the pipeline initializes correctly."""                                                                                                                               
    config = OmegaConf.create({                                                                                                                                                       
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["fine_tuning", "evaluation"],                                                                                                                                  
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
    })                                                                                                                                                                                
                                                                                                                                                                                    
    pipeline = OptimizationPipeline(config)                                                                                                                                           
                                                                                                                                                                                    
    assert len(pipeline.stages) == 2                                                                                                                                                  
    assert pipeline.config.model.name == "gpt2"                                                                                                                                       
    assert pipeline.config.pipeline.stages == ["fine_tuning", "evaluation"]                                                                                                           
                                                                                                                                                                                    
                                                                                                                                                                                    
def test_pipeline_with_invalid_stage():                                                                                                                                               
    """Test that the pipeline raises an error with an invalid stage."""                                                                                                               
    config = OmegaConf.create({                                                                                                                                                       
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["invalid_stage"],                                                                                                                                              
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
    })                                                                                                                                                                                
                                                                                                                                                                                    
    with pytest.raises(ValueError):                                                                                                                                                   
        pipeline = OptimizationPipeline(config)                                                                                                                                       
                                                                                                                                                                                    
                                                                                                                                                                                    
def test_pipeline_run_with_mock_stages(monkeypatch):                                                                                                                                  
    """Test that the pipeline runs correctly with mock stages."""                                                                                                                     
    # Register mock stage                                                                                                                                                             
    monkeypatch.setitem(OptimizationPipeline.STAGE_REGISTRY, "mock_stage", MockStage)                                                                                                 
                                                                                                                                                                                    
    config = OmegaConf.create({                                                                                                                                                       
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["mock_stage"],                                                                                                                                                 
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
    })                                                                                                                                                                                
                                                                                                                                                                                    
    pipeline = OptimizationPipeline(config)                                                                                                                                           
    results = pipeline.run()                                                                                                                                                          
                                                                                                                                                                                    
    assert "completed_stages" in results                                                                                                                                              
    assert "metrics" in results                                                                                                                                                       
    assert "artifacts" in results                                                                                                                                                     
    assert results["completed_stages"] == ["mock_stage"]                                                                                                                              
    assert "mock_stage" in results["metrics"]                                                                                                                                         
    assert results["metrics"]["mock_stage"]["mock_metric"] == 0.95                                                                                                                    
                                                                                                                                                                                    
                                                                                                                                                                                    
def test_get_stage():                                                                                                                                                                 
    """Test getting a stage by name."""                                                                                                                                               
    # Register mock stage                                                                                                                                                             
    OptimizationPipeline.STAGE_REGISTRY["mock_stage"] = MockStage                                                                                                                     
                                                                                                                                                                                    
    config = OmegaConf.create({                                                                                                                                                       
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["mock_stage"],                                                                                                                                                 
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
    })                                                                                                                                                                                
                                                                                                                                                                                    
    pipeline = OptimizationPipeline(config)                                                                                                                                           
    stage = pipeline.get_stage("mock_stage")                                                                                                                                          
                                                                                                                                                                                    
    assert stage is not None                                                                                                                                                          
    assert isinstance(stage, MockStage)                                                                                                                                               
                                                                                                                                                                                    
    # Test getting a non-existent stage                                                                                                                                               
    assert pipeline.get_stage("non_existent") is None 