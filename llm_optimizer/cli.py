"""                                                                                                                                                                                   
Command-line interface for the LLM Optimizer.                                                                                                                                         
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import sys                                                                                                                                                                            
from pathlib import Path                                                                                                                                                              
from typing import Optional, List                                                                                                                                                     
                                                                                                                                                                                    
import typer                                                                                                                                                                          
from rich.console import Console                                                                                                                                                      
from rich.panel import Panel                                                                                                                                                          
from omegaconf import OmegaConf                                                                                                                                                       
                                                                                                                                                                                    
from llm_optimizer.config import load_config                                                                                                                                          
from llm_optimizer.pipeline import OptimizationPipeline                                                                                                                               
                                                                                                                                                                                    
app = typer.Typer(                                                                                                                                                                    
    name="llm-optimizer",                                                                                                                                                             
    help="A modular pipeline for LLM post-training optimization",                                                                                                                     
    add_completion=False,                                                                                                                                                             
)                                                                                                                                                                                     
console = Console()                                                                                                                                                                   
                                                                                                                                                                                    
                                                                                                                                                                                    
@app.command()                                                                                                                                                                        
def optimize(                                                                                                                                                                         
    config_path: str = typer.Argument(                                                                                                                                                
        ..., help="Path to the configuration file or directory"                                                                                                                       
    ),                                                                                                                                                                                
    output_dir: str = typer.Option(                                                                                                                                                   
        "./outputs", help="Directory to save optimization outputs"                                                                                                                    
    ),                                                                                                                                                                                
    stages: Optional[List[str]] = typer.Option(                                                                                                                                       
        None, help="Specific pipeline stages to run (comma-separated)"                                                                                                                
    ),                                                                                                                                                                                
    debug: bool = typer.Option(False, help="Enable debug mode"),                                                                                                                      
):                                                                                                                                                                                    
    """                                                                                                                                                                               
    Run the LLM optimization pipeline with the specified configuration.                                                                                                               
    """                                                                                                                                                                               
    try:                                                                                                                                                                              
        # Load configuration                                                                                                                                                          
        config = load_config(config_path)                                                                                                                                             
                                                                                                                                                                                    
        # Override config with CLI options                                                                                                                                            
        if output_dir:                                                                                                                                                                
            config.output_dir = output_dir                                                                                                                                            
                                                                                                                                                                                    
        if stages:                                                                                                                                                                    
            config.pipeline.stages = stages                                                                                                                                           
                                                                                                                                                                                    
        if debug:                                                                                                                                                                     
            config.debug = True                                                                                                                                                       
                                                                                                                                                                                    
        # Create output directory                                                                                                                                                     
        os.makedirs(config.output_dir, exist_ok=True)                                                                                                                                 
                                                                                                                                                                                    
        # Save the resolved configuration                                                                                                                                             
        config_save_path = Path(config.output_dir) / "config.yaml"                                                                                                                    
        with open(config_save_path, "w") as f:                                                                                                                                        
            f.write(OmegaConf.to_yaml(config))                                                                                                                                        
                                                                                                                                                                                    
        # Display configuration summary                                                                                                                                               
        console.print(                                                                                                                                                                
            Panel.fit(                                                                                                                                                                
                f"[bold green]LLM Optimizer[/bold green]\n\n"                                                                                                                         
                f"Configuration: {config_path}\n"                                                                                                                                     
                f"Output directory: {config.output_dir}\n"                                                                                                                            
                f"Pipeline stages: {', '.join(config.pipeline.stages)}\n"                                                                                                             
                f"Model: {config.model.name}"                                                                                                                                         
            )                                                                                                                                                                         
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Initialize and run the pipeline                                                                                                                                             
        pipeline = OptimizationPipeline(config)                                                                                                                                       
        results = pipeline.run()                                                                                                                                                      
                                                                                                                                                                                    
        # Display results summary                                                                                                                                                     
        console.print(                                                                                                                                                                
            Panel.fit(                                                                                                                                                                
                f"[bold green]Optimization Complete[/bold green]\n\n"                                                                                                                 
                f"Results saved to: {config.output_dir}\n"                                                                                                                            
                f"Stages completed: {', '.join(results['completed_stages'])}\n"                                                                                                       
                f"Total time: {results['total_time']:.2f} seconds"                                                                                                                    
            )                                                                                                                                                                         
        )                                                                                                                                                                             
                                                                                                                                                                                    
        return 0                                                                                                                                                                      
                                                                                                                                                                                    
    except Exception as e:                                                                                                                                                            
        console.print(f"[bold red]Error:[/bold red] {str(e)}")                                                                                                                        
        if debug:                                                                                                                                                                     
            console.print_exception()                                                                                                                                                 
        return 1                                                                                                                                                                      
                                                                                                                                                                                    
                                                                                                                                                                                    
@app.command()
def benchmark(
    config_path: str = typer.Option(                                                                                                                                                  
         ..., "--config", "-c", help="Path to the benchmarking configuration file"                                                                                                     
     ),
    baseline_model: Optional[str] = typer.Option(
        None, help="Path or name of the baseline model (overrides config)"
    ),
    optimized_model: Optional[str] = typer.Option(
        None, help="Path or name of the optimized model (overrides config)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Directory to save benchmark results (overrides config)"
    ),
    no_quality: bool = typer.Option(
        False, help="Skip quality benchmarking (faster)"
    ),
    performance_only: bool = typer.Option(
        False, help="Run only performance benchmarks (no quality or memory tests)"
    ),
    sequence_lengths: Optional[str] = typer.Option(
        None, help="Comma-separated sequence lengths to test (e.g., '128,256,512')"
    ),
    batch_sizes: Optional[str] = typer.Option(
        None, help="Comma-separated batch sizes to test (e.g., '1,2,4')"
    ),
    iterations: Optional[int] = typer.Option(
        None, help="Number of iterations for each benchmark test"
    ),
    debug: bool = typer.Option(False, help="Enable debug mode"),
):
    """
    Benchmark models to compare performance, memory usage, and quality.
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Override config with CLI options
        if baseline_model:
            config.benchmark.baseline_model = baseline_model
            
        if optimized_model:
            config.benchmark.optimized_model = optimized_model
            
        if output_dir:
            config.benchmark.output_dir = output_dir
            
        if no_quality:
            config.benchmark.benchmark_quality = False
            
        if performance_only:
            config.benchmark.benchmark_quality = False
            config.benchmark.benchmark_memory = False
            
        if sequence_lengths:
            try:
                config.benchmark.sequence_lengths = [int(x.strip()) for x in sequence_lengths.split(',')]
            except ValueError:
                console.print("[bold red]Error:[/bold red] Invalid sequence lengths format. Use comma-separated integers.")
                return 1
                
        if batch_sizes:
            try:
                config.benchmark.batch_sizes = [int(x.strip()) for x in batch_sizes.split(',')]
            except ValueError:
                console.print("[bold red]Error:[/bold red] Invalid batch sizes format. Use comma-separated integers.")
                return 1
                
        if iterations:
            config.benchmark.num_iterations = iterations
            
        if debug:
            config.debug = True
            
        # Create output directory
        os.makedirs(config.benchmark.output_dir, exist_ok=True)
        
        # Save the resolved configuration
        config_save_path = Path(config.benchmark.output_dir) / "benchmark_config.yaml"
        with open(config_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(config))
            
        # Display configuration summary
        console.print(
            Panel.fit(
                f"[bold green]LLM Benchmarking[/bold green]\n\n"
                f"Baseline model: {config.benchmark.baseline_model}\n"
                f"Optimized model: {config.benchmark.optimized_model}\n"
                f"Output directory: {config.benchmark.output_dir}\n"
                f"Quality benchmarking: {'Enabled' if config.benchmark.get('benchmark_quality', True) else 'Disabled'}"
            )
        )
        
        # Initialize and run the benchmarking stage
        from llm_optimizer.benchmarking import BenchmarkingStage
        
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
            Panel.fit(
                f"[bold green]Benchmarking Complete[/bold green]\n\n"
                f"Results saved to: {config.benchmark.output_dir}\n"
                f"See benchmark_summary.txt for detailed results"
            )
        )
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if debug:
            console.print_exception()
        return 1


@app.command()                                                                                                                                                                        
def evaluate(                                                                                                                                                                         
    model_path: str = typer.Argument(..., help="Path to the optimized model"),                                                                                                        
    eval_config: str = typer.Argument(..., help="Path to evaluation configuration"),                                                                                                  
    output_file: Optional[str] = typer.Option(                                                                                                                                        
        None, help="Path to save evaluation results"                                                                                                                                  
    ),                                                                                                                                                                                
):                                                                                                                                                                                    
    """                                                                                                                                                                               
    Evaluate an optimized model using specified metrics.                                                                                                                              
    """                                                                                                                                                                               
    try:                                                                                                                                                                              
        from llm_optimizer.evaluation import ModelEvaluator                                                                                                                           
                                                                                                                                                                                    
        # Load evaluation configuration                                                                                                                                               
        config = load_config(eval_config)                                                                                                                                             
                                                                                                                                                                                    
        console.print(                                                                                                                                                                
            Panel.fit(                                                                                                                                                                
                f"[bold green]Model Evaluation[/bold green]\n\n"                                                                                                                      
                f"Model: {model_path}\n"                                                                                                                                              
                f"Evaluation config: {eval_config}"                                                                                                                                   
            )                                                                                                                                                                         
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Run evaluation                                                                                                                                                              
        evaluator = ModelEvaluator(config)                                                                                                                                            
        results = evaluator.evaluate(model_path)                                                                                                                                      
                                                                                                                                                                                    
        # Save results if output file is specified                                                                                                                                    
        if output_file:                                                                                                                                                               
            import json                                                                                                                                                               
            with open(output_file, "w") as f:                                                                                                                                         
                json.dump(results, f, indent=2)                                                                                                                                       
            console.print(f"Results saved to: {output_file}")                                                                                                                         
                                                                                                                                                                                    
        # Display results                                                                                                                                                             
        console.print("\n[bold]Evaluation Results:[/bold]")                                                                                                                           
        for metric, value in results.items():                                                                                                                                         
            console.print(f"{metric}: {value}")                                                                                                                                       
                                                                                                                                                                                    
        return 0                                                                                                                                                                      
                                                                                                                                                                                    
    except Exception as e:                                                                                                                                                            
        console.print(f"[bold red]Error:[/bold red] {str(e)}")                                                                                                                        
        return 1                                                                                                                                                                      
                                                                                                                                                                                    
                                                                                                                                                                                    
@app.command()                                                                                                                                                                        
def init(                                                                                                                                                                             
    project_dir: str = typer.Argument("./llm_project", help="Project directory to create"),                                                                                           
    template: str = typer.Option(                                                                                                                                                     
        "basic", help="Template to use (basic, full, or minimal)"                                                                                                                     
    ),                                                                                                                                                                                
):                                                                                                                                                                                    
    """                                                                                                                                                                               
    Initialize a new LLM optimization project with template files.                                                                                                                    
    """                                                                                                                                                                               
    from llm_optimizer.utils.project import create_project_template                                                                                                                   
                                                                                                                                                                                    
    try:                                                                                                                                                                              
        project_path = create_project_template(project_dir, template)                                                                                                                 
        console.print(                                                                                                                                                                
            f"[bold green]Project initialized at:[/bold green] {project_path}\n"                                                                                                      
            f"Template: {template}\n\n"                                                                                                                                               
            f"To get started, run:\n"                                                                                                                                                 
            f"cd {project_dir}\n"                                                                                                                                                     
            f"llm-optimizer optimize configs/default.yaml"                                                                                                                            
        )                                                                                                                                                                             
        return 0                                                                                                                                                                      
    except Exception as e:                                                                                                                                                            
        console.print(f"[bold red]Error:[/bold red] {str(e)}")                                                                                                                        
        return 1                                                                                                                                                                      
                                                                                                                                                                                    
                                                                                                                                                                                    
def main():                                                                                                                                                                           
    """Entry point for the CLI."""                                                                                                                                                    
    return app()                                                                                                                                                                      
                                                                                                                                                                                    
                                                                                                                                                                                    
if __name__ == "__main__":                                                                                                                                                            
    sys.exit(main())  
