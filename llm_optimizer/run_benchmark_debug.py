"""
Script to run benchmarking with enhanced diagnostics.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

from omegaconf import OmegaConf
from rich.console import Console

from llm_optimizer.benchmarking import BenchmarkingStage
from llm_optimizer.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

def run_benchmark_debug(config_path: str) -> Dict[str, Any]:
    """
    Run benchmarking with enhanced diagnostics.
    
    Args:
        config_path: Path to the benchmark configuration
        
    Returns:
        Dictionary with benchmark results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Enable debug mode
    config.debug = True
    
    # Create output directory
    os.makedirs(config.benchmark.output_dir, exist_ok=True)
    
    # Save the resolved configuration
    config_save_path = os.path.join(config.benchmark.output_dir, "benchmark_debug_config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))
    
    # Display configuration summary
    console.print(
        f"[bold green]LLM Benchmarking (Debug Mode)[/bold green]\n\n"
        f"Baseline model: {config.benchmark.baseline_model}\n"
        f"Optimized model: {config.benchmark.optimized_model}\n"
        f"Output directory: {config.benchmark.output_dir}\n"
        f"Debug mode: Enabled"
    )
    
    # Create a minimal model state for the optimized model
    model_state = {
        "model_path": config.benchmark.optimized_model,
        "is_pretrained": True,
    }
    
    # Run benchmarking
    benchmark_stage = BenchmarkingStage(config.benchmark)
    results = benchmark_stage.run(model_state)
    
    # Display results summary
    console.print(
        f"[bold green]Benchmarking Complete[/bold green]\n\n"
        f"Results saved to: {config.benchmark.output_dir}\n"
        f"See benchmark_debug.log for detailed diagnostics"
    )
    
    return results

def main():
    """Main function for the benchmark debug script."""
    parser = argparse.ArgumentParser(description="Run benchmarking with enhanced diagnostics")
    parser.add_argument("--config", type=str, required=True, help="Path to the benchmark configuration")
    
    args = parser.parse_args()
    
    try:
        results = run_benchmark_debug(args.config)
        return 0
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
