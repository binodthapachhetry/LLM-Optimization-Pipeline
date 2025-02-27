"""                                                                                                                                                                                   
Project utility functions for LLM optimization.                                                                                                                                       
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import shutil                                                                                                                                                                         
import logging                                                                                                                                                                        
from pathlib import Path                                                                                                                                                              
from typing import Dict, Any, Optional, List                                                                                                                                          
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
def create_project_template(                                                                                                                                                          
    project_dir: str,                                                                                                                                                                 
    template: str = "basic",                                                                                                                                                          
) -> str:                                                                                                                                                                             
    """                                                                                                                                                                               
    Create a new LLM optimization project with template files.                                                                                                                        
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        project_dir: Directory to create the project in                                                                                                                               
        template: Template to use (basic, full, or minimal)                                                                                                                           
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Path to the created project                                                                                                                                                   
    """                                                                                                                                                                               
    project_path = Path(project_dir)                                                                                                                                                  
                                                                                                                                                                                    
    # Create project directory                                                                                                                                                        
    project_path.mkdir(parents=True, exist_ok=True)                                                                                                                                   
                                                                                                                                                                                    
    # Create directory structure                                                                                                                                                      
    dirs = [                                                                                                                                                                          
        "configs",                                                                                                                                                                    
        "data",                                                                                                                                                                       
        "outputs",                                                                                                                                                                    
        "scripts",                                                                                                                                                                    
        "tests",                                                                                                                                                                      
    ]                                                                                                                                                                                 
                                                                                                                                                                                    
    for dir_name in dirs:                                                                                                                                                             
        (project_path / dir_name).mkdir(exist_ok=True)                                                                                                                                
                                                                                                                                                                                    
    # Create config files based on template                                                                                                                                           
    if template == "minimal":                                                                                                                                                         
        _create_minimal_template(project_path)                                                                                                                                        
    elif template == "full":                                                                                                                                                          
        _create_full_template(project_path)                                                                                                                                           
    else:  # basic (default)                                                                                                                                                          
        _create_basic_template(project_path)                                                                                                                                          
                                                                                                                                                                                    
    # Create README                                                                                                                                                                   
    _create_readme(project_path, template)                                                                                                                                            
                                                                                                                                                                                    
    return str(project_path.absolute())                                                                                                                                               
                                                                                                                                                                                    
                                                                                                                                                                                    
def _create_minimal_template(project_path: Path) -> None:                                                                                                                             
    """Create a minimal project template."""                                                                                                                                          
    # Create minimal config                                                                                                                                                           
    config = {                                                                                                                                                                        
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["evaluation"],                                                                                                                                                 
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    _write_yaml(project_path / "configs" / "default.yaml", config)                                                                                                                    
                                                                                                                                                                                    
                                                                                                                                                                                    
def _create_basic_template(project_path: Path) -> None:                                                                                                                               
    """Create a basic project template."""                                                                                                                                            
    # Create basic configs                                                                                                                                                            
    default_config = {                                                                                                                                                                
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["fine_tuning", "evaluation"],                                                                                                                                  
        },                                                                                                                                                                            
        "fine_tuning": {                                                                                                                                                              
            "method": "lora",                                                                                                                                                         
            "dataset": "wikitext",                                                                                                                                                    
            "dataset_config": "wikitext-2-raw-v1",                                                                                                                                    
            "num_epochs": 3,                                                                                                                                                          
            "batch_size": 4,                                                                                                                                                          
            "learning_rate": 5e-5,                                                                                                                                                    
        },                                                                                                                                                                            
        "evaluation": {                                                                                                                                                               
            "evaluate_perplexity": True,                                                                                                                                              
            "evaluate_tasks": True,                                                                                                                                                   
            "tasks": ["lambada"],                                                                                                                                                     
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
        "debug": False,                                                                                                                                                               
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    quantization_config = {                                                                                                                                                           
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["quantization", "evaluation", "benchmarking"],                                                                                                                 
        },                                                                                                                                                                            
        "quantization": {                                                                                                                                                             
            "method": "int8",                                                                                                                                                         
        },                                                                                                                                                                            
        "evaluation": {                                                                                                                                                               
            "evaluate_perplexity": True,                                                                                                                                              
            "evaluate_tasks": True,                                                                                                                                                   
            "tasks": ["lambada"],                                                                                                                                                     
        },                                                                                                                                                                            
        "benchmarking": {                                                                                                                                                             
            "baseline_model": "gpt2",                                                                                                                                                 
            "sequence_lengths": [128, 512],                                                                                                                                           
            "batch_sizes": [1, 4],                                                                                                                                                    
        },                                                                                                                                                                            
        "output_dir": "./outputs/quantized",                                                                                                                                          
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    _write_yaml(project_path / "configs" / "default.yaml", default_config)                                                                                                            
    _write_yaml(project_path / "configs" / "quantization.yaml", quantization_config)                                                                                                  
                                                                                                                                                                                    
    # Create example script                                                                                                                                                           
    example_script = """#!/usr/bin/env python                                                                                                                                         
# Example script to run the LLM optimization pipeline                                                                                                                                 
                                                                                                                                                                                    
import os                                                                                                                                                                             
import argparse                                                                                                                                                                       
from llm_optimizer.cli import main                                                                                                                                                    
                                                                                                                                                                                    
if __name__ == "__main__":                                                                                                                                                            
    # This script is a wrapper around the CLI                                                                                                                                         
    # You can customize it for your specific needs                                                                                                                                    
    main()                                                                                                                                                                            
"""                                                                                                                                                                                   
                                                                                                                                                                                    
    with open(project_path / "scripts" / "run_pipeline.py", "w") as f:                                                                                                                
        f.write(example_script)                                                                                                                                                       
                                                                                                                                                                                    
    # Make script executable                                                                                                                                                          
    os.chmod(project_path / "scripts" / "run_pipeline.py", 0o755)                                                                                                                     
                                                                                                                                                                                    
                                                                                                                                                                                    
def _create_full_template(project_path: Path) -> None:                                                                                                                                
    """Create a full project template with all components."""                                                                                                                         
    # Create the basic template first                                                                                                                                                 
    _create_basic_template(project_path)                                                                                                                                              
                                                                                                                                                                                    
    # Add more advanced configs                                                                                                                                                       
    distillation_config = {                                                                                                                                                           
        "model": {                                                                                                                                                                    
            "name": "gpt2-large",  # Teacher model                                                                                                                                    
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["distillation", "evaluation", "benchmarking"],                                                                                                                 
        },                                                                                                                                                                            
        "distillation": {                                                                                                                                                             
            "student_model": "distilgpt2",                                                                                                                                            
            "temperature": 2.0,                                                                                                                                                       
            "alpha": 0.5,                                                                                                                                                             
            "dataset": "wikitext",                                                                                                                                                    
            "dataset_config": "wikitext-2-raw-v1",                                                                                                                                    
            "num_epochs": 3,                                                                                                                                                          
        },                                                                                                                                                                            
        "evaluation": {                                                                                                                                                               
            "evaluate_perplexity": True,                                                                                                                                              
            "evaluate_tasks": True,                                                                                                                                                   
            "tasks": ["lambada", "hellaswag"],                                                                                                                                        
        },                                                                                                                                                                            
        "benchmarking": {                                                                                                                                                             
            "baseline_model": "gpt2-large",                                                                                                                                           
            "sequence_lengths": [128, 512, 1024],                                                                                                                                     
            "batch_sizes": [1, 4, 8],                                                                                                                                                 
            "benchmark_quality": True,                                                                                                                                                
        },                                                                                                                                                                            
        "output_dir": "./outputs/distilled",                                                                                                                                          
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    pruning_config = {                                                                                                                                                                
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["pruning", "evaluation", "benchmarking"],                                                                                                                      
        },                                                                                                                                                                            
        "pruning": {                                                                                                                                                                  
            "method": "magnitude",                                                                                                                                                    
            "amount": 0.3,                                                                                                                                                            
        },                                                                                                                                                                            
        "evaluation": {                                                                                                                                                               
            "evaluate_perplexity": True,                                                                                                                                              
            "evaluate_tasks": True,                                                                                                                                                   
            "tasks": ["lambada"],                                                                                                                                                     
        },                                                                                                                                                                            
        "benchmarking": {                                                                                                                                                             
            "baseline_model": "gpt2",                                                                                                                                                 
            "sequence_lengths": [128, 512],                                                                                                                                           
            "batch_sizes": [1, 4],                                                                                                                                                    
        },                                                                                                                                                                            
        "output_dir": "./outputs/pruned",                                                                                                                                             
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    onnx_config = {                                                                                                                                                                   
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["onnx_conversion", "benchmarking"],                                                                                                                            
        },                                                                                                                                                                            
        "onnx_conversion": {                                                                                                                                                          
            "opset_version": 13,                                                                                                                                                      
        },                                                                                                                                                                            
        "benchmarking": {                                                                                                                                                             
            "baseline_model": "gpt2",                                                                                                                                                 
            "sequence_lengths": [128, 512],                                                                                                                                           
            "batch_sizes": [1, 4],                                                                                                                                                    
        },                                                                                                                                                                            
        "output_dir": "./outputs/onnx",                                                                                                                                               
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    full_pipeline_config = {                                                                                                                                                          
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": [                                                                                                                                                               
                "fine_tuning",                                                                                                                                                        
                "quantization",                                                                                                                                                       
                "evaluation",                                                                                                                                                         
                "onnx_conversion",                                                                                                                                                    
                "benchmarking"                                                                                                                                                        
            ],                                                                                                                                                                        
        },                                                                                                                                                                            
        "fine_tuning": {                                                                                                                                                              
            "method": "lora",                                                                                                                                                         
            "dataset": "wikitext",                                                                                                                                                    
            "dataset_config": "wikitext-2-raw-v1",                                                                                                                                    
            "num_epochs": 3,                                                                                                                                                          
        },                                                                                                                                                                            
        "quantization": {                                                                                                                                                             
            "method": "int8",                                                                                                                                                         
        },                                                                                                                                                                            
        "evaluation": {                                                                                                                                                               
            "evaluate_perplexity": True,                                                                                                                                              
            "evaluate_tasks": True,                                                                                                                                                   
            "tasks": ["lambada"],                                                                                                                                                     
        },                                                                                                                                                                            
        "onnx_conversion": {                                                                                                                                                          
            "opset_version": 13,                                                                                                                                                      
        },                                                                                                                                                                            
        "benchmarking": {                                                                                                                                                             
            "baseline_model": "gpt2",                                                                                                                                                 
            "sequence_lengths": [128, 512],                                                                                                                                           
            "batch_sizes": [1, 4],                                                                                                                                                    
        },                                                                                                                                                                            
        "output_dir": "./outputs/full_pipeline",                                                                                                                                      
    }                                                                                                                                                                                 
                                                                                                                                                                                    
    _write_yaml(project_path / "configs" / "distillation.yaml", distillation_config)                                                                                                  
    _write_yaml(project_path / "configs" / "pruning.yaml", pruning_config)                                                                                                            
    _write_yaml(project_path / "configs" / "onnx.yaml", onnx_config)                                                                                                                  
    _write_yaml(project_path / "configs" / "full_pipeline.yaml", full_pipeline_config)                                                                                                
                                                                                                                                                                                    
    # Create example test                                                                                                                                                             
    test_file = """import pytest                                                                                                                                                      
from llm_optimizer.pipeline import OptimizationPipeline                                                                                                                               
from omegaconf import OmegaConf                                                                                                                                                       
                                                                                                                                                                                    
def test_pipeline_initialization():                                                                                                                                                   
    #Test that the pipeline initializes correctly                                                                                                                              
    config = OmegaConf.create({                                                                                                                                                       
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["evaluation"],                                                                                                                                                 
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
    })                                                                                                                                                                                
                                                                                                                                                                                    
    pipeline = OptimizationPipeline(config)                                                                                                                                           
    assert len(pipeline.stages) == 1                                                                                                                                                  
    assert pipeline.config.model.name == "gpt2"                                                                                                                                       
"""                                                                                                                                                                                   
                                                                                                                                                                                    
    with open(project_path / "tests" / "test_pipeline.py", "w") as f:                                                                                                                 
        f.write(test_file)                                                                                                                                                            
                                                                                                                                                                                    
                                                                                                                                                                                    
def _create_readme(project_path: Path, template: str) -> None:                                                                                                                        
    """Create a README file for the project."""                                                                                                                                       
    readme_content = f"""# LLM Optimization Project                                                                                                                                   
                                                                                                                                                                                    
This project was created using the LLM Optimizer framework with the '{template}' template.                                                                                            
                                                                                                                                                                                    
## Getting Started                                                                                                                                                                    
                                                                                                                                                                                    
1. Install the required dependencies:                                                                                                                                                 
                                                                                                                                                                                    

pip install -r requirements.txt                                                                                                                                                        

                                                                                                                                                                                    
                                                                                                                                                                                    
2. Run the optimization pipeline:                                                                                                                                                     
                                                                                                                                                                                    

llm-optimizer optimize configs/default.yaml                                                                                                                                            

                                                                                                                                                                                    
                                                                                                                                                                                    
## Project Structure                                                                                                                                                                  
                                                                                                                                                                                    
- `configs/`: Configuration files for different optimization strategies                                                                                                               
- `data/`: Directory for datasets                                                                                                                                                     
- `outputs/`: Output directory for optimized models and results                                                                                                                       
- `scripts/`: Utility scripts                                                                                                                                                         
- `tests/`: Test files                                                                                                                                                                
                                                                                                                                                                                    
## Available Configurations                                                                                                                                                           
                                                                                                                                                                                    
- `default.yaml`: Default configuration for fine-tuning and evaluation                                                                                                                
"""                                                                                                                                                                                   
                                                                                                                                                                                    
if template == "full":                                                                                                                                                               
    readme_content += """                                                                                                                                                            
- `quantization.yaml`: Configuration for quantizing models                                                                                                                            
- `distillation.yaml`: Configuration for knowledge distillation                                                                                                                       
- `pruning.yaml`: Configuration for model pruning                                                                                                                                     
- `onnx.yaml`: Configuration for ONNX conversion                                                                                                                                      
- `full_pipeline.yaml`: End-to-end optimization pipeline                                                                                                                              
"""                                                                                                                                                                                   
                                                                                                                                                                                    
readme_content += """                                                                                                                                                                
## Customization                                                                                                                                                                      
                                                                                                                                                                                    
You can customize the configurations in the `configs/` directory to suit your specific needs.                                                                                         
                                                                                                                                                                                    
## License                                                                                                                                                                            
                                                                                                                                                                                    
This project is licensed under the MIT License.                                                                                                                                       
"""                                                                                                                                                                                   
                                                                                                                                                                                    
with open(project_path / "README.md", "w") as f:                                                                                                                                     
    f.write(readme_content)                                                                                                                                                          
                                                                                                                                                                                    
                                                                                                                                                                                    
def _write_yaml(file_path: Path, data: Dict) -> None:                                                                                                                                 
    #Write data to a YAML file                                                                                                                                                    
    try:                                                                                                                                                                                 
        import yaml                                                                                                                                                                      
                                                                                                                                                                                        
        with open(file_path, "w") as f:                                                                                                                                                  
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)                                                                                                                
    except ImportError:                                                                                                                                                                  
        # Fallback if PyYAML is not available                                                                                                                                            
        import json                                                                                                                                                                      
                                                                                                                                                                                        
        with open(file_path, "w") as f:                                                                                                                                                  
            json.dump(data, f, indent=2) 