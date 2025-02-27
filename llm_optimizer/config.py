"""                                                                                                                                                                                   
Configuration management for the LLM Optimizer.                                                                                                                                       
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
from pathlib import Path                                                                                                                                                              
from typing import Dict, Any, Union, Optional                                                                                                                                         
                                                                                                                                                                                    
import hydra                                                                                                                                                                          
from hydra.core.config_store import ConfigStore                                                                                                                                       
from omegaconf import OmegaConf, DictConfig                                                                                                                                           
                                                                                                                                                                                    
# Default configuration directory                                                                                                                                                     
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")                                                                                                                       
                                                                                                                                                                                    
                                                                                                                                                                                    
def register_configs() -> None:                                                                                                                                                       
    """Register configuration schemas with Hydra."""                                                                                                                                  
    cs = ConfigStore.instance()                                                                                                                                                       
                                                                                                                                                                                    
    # Register configuration schemas here                                                                                                                                             
    # This allows for validation and auto-completion                                                                                                                                  
    # cs.store(name="config_schema", node=ConfigSchema)                                                                                                                               
                                                                                                                                                                                    
                                                                                                                                                                                    
def load_config(config_path: str) -> DictConfig:                                                                                                                                      
    """                                                                                                                                                                               
    Load configuration from a file or directory.                                                                                                                                      
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        config_path: Path to configuration file or directory                                                                                                                          
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Loaded configuration as DictConfig                                                                                                                                            
    """                                                                                                                                                                               
    # If config_path is a directory, use Hydra to compose config                                                                                                                      
    if os.path.isdir(config_path):                                                                                                                                                    
        hydra.initialize(config_path=config_path)                                                                                                                                     
        config = hydra.compose(config_name="config")                                                                                                                                  
        return config                                                                                                                                                                 
                                                                                                                                                                                    
    # If config_path is a file, load it directly                                                                                                                                      
    if os.path.isfile(config_path):                                                                                                                                                   
        return OmegaConf.load(config_path)                                                                                                                                            
                                                                                                                                                                                    
    # If config_path is a name without extension, try to find it                                                                                                                      
    if "." not in os.path.basename(config_path):                                                                                                                                      
        # Try with yaml extension                                                                                                                                                     
        yaml_path = f"{config_path}.yaml"                                                                                                                                             
        if os.path.isfile(yaml_path):                                                                                                                                                 
            return OmegaConf.load(yaml_path)                                                                                                                                          
                                                                                                                                                                                    
        # Try in default config directory                                                                                                                                             
        default_path = os.path.join(CONFIG_DIR, f"{config_path}.yaml")                                                                                                                
        if os.path.isfile(default_path):                                                                                                                                              
            return OmegaConf.load(default_path)                                                                                                                                       
                                                                                                                                                                                    
    raise FileNotFoundError(f"Configuration not found: {config_path}")                                                                                                                
                                                                                                                                                                                    
                                                                                                                                                                                    
def save_config(config: DictConfig, output_path: str) -> None:                                                                                                                        
    """                                                                                                                                                                               
    Save configuration to a file.                                                                                                                                                     
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        config: Configuration to save                                                                                                                                                 
        output_path: Path to save the configuration                                                                                                                                   
    """                                                                                                                                                                               
    with open(output_path, "w") as f:                                                                                                                                                 
        f.write(OmegaConf.to_yaml(config))                                                                                                                                            
                                                                                                                                                                                    
                                                                                                                                                                                    
def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:                                                                                                
    """                                                                                                                                                                               
    Merge two configurations, with override_config taking precedence.                                                                                                                 
                                                                                                                                                                                    
    Args:                                                                                                                                                                             
        base_config: Base configuration                                                                                                                                               
        override_config: Configuration to override base with                                                                                                                          
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Merged configuration                                                                                                                                                          
    """                                                                                                                                                                               
    return OmegaConf.merge(base_config, override_config)                                                                                                                              
                                                                                                                                                                                    
                                                                                                                                                                                    
def get_default_config() -> DictConfig:                                                                                                                                               
    """                                                                                                                                                                               
    Get the default configuration.                                                                                                                                                    
                                                                                                                                                                                    
    Returns:                                                                                                                                                                          
        Default configuration                                                                                                                                                         
    """                                                                                                                                                                               
    default_config_path = os.path.join(CONFIG_DIR, "default.yaml")                                                                                                                    
    if os.path.exists(default_config_path):                                                                                                                                           
        return OmegaConf.load(default_config_path)                                                                                                                                    
                                                                                                                                                                                    
    # If default config doesn't exist, return a basic config                                                                                                                          
    return OmegaConf.create({                                                                                                                                                         
        "model": {                                                                                                                                                                    
            "name": "gpt2",                                                                                                                                                           
            "pretrained": True,                                                                                                                                                       
        },                                                                                                                                                                            
        "pipeline": {                                                                                                                                                                 
            "stages": ["fine_tuning", "evaluation"],                                                                                                                                  
        },                                                                                                                                                                            
        "output_dir": "./outputs",                                                                                                                                                    
        "debug": False,                                                                                                                                                               
    })                     