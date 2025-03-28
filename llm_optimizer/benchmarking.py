"""                                                                                                                                                                                   
Benchmarking module for LLM optimization.                                                                                                                                             
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import time                                                                                                                                                                           
import json                                                                                                                                                                           
import logging                                                                                                                                                                        
from typing import Dict, Any, List, Optional, Tuple                                                                                                                                   
                                                                                                                                                                                    
import torch                                                                                                                                                                          
import numpy as np                                                                                                                                                                    
import pandas as pd                                                                                                                                                                   
import matplotlib.pyplot as plt                                                                                                                                                       
import seaborn as sns                                                                                                                                                                 
from transformers import AutoTokenizer

from llm_optimizer.utils.diagnostics import log_completion_evaluation, test_model_completion, analyze_tokenizer_behavior
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage                                                                                                                                      
from llm_optimizer.utils.model import load_model_and_tokenizer
from llm_optimizer.utils.gguf_utils import load_gguf_model

from rich.console import Console  
# from omegaconf import OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console() 

                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
class BenchmarkingStage(OptimizationStage):                                                                                                                                           
    """                                                                                                                                                                               
    Benchmarking stage for LLM optimization.                                                                                                                                          
                                                                                                                                                                                    
    Compares optimized models against baselines on various metrics:                                                                                                                   
    - Inference speed                                                                                                                                                                 
    - Memory usage                                                                                                                                                                    
    - Accuracy                                                                                                                                                                        
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                     
        """                                                                                                                                                                           
        Run the benchmarking stage.                                                                                                                                                   
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_state: Current state of the model                                                                                                                                   
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Benchmarking results                                                                                                                                                      
        """    
        try:
            self.validate_input(model_state)    
                                                                                                                                                                                    
            # Extract configuration                                                                                                                                                       
            optimized_model_path = model_state["model_path"]                                                                                                                              
            baseline_model_path = self.config.get("baseline_model", "gpt2")                                                                                                               
            output_dir = os.path.join(                                                                                                                                                    
                self.config.get("output_dir", "./outputs"),                                                                                                                               
                "benchmarks"                                                                                                                                                              
            )                                                                                                                                                                             
                                                                                                                                                                                    
            logger.info(f"Benchmarking optimized model {optimized_model_path} against baseline {baseline_model_path}")
            console.print(f"[bold blue]Loading models...[/bold blue]")
                                                                                                                                                                                    
            # Load models with format detection - with error handling
            optimized_model = optimized_tokenizer = None
            baseline_model = baseline_tokenizer = None
            
            try:
                console.print(f"Loading optimized model: {optimized_model_path}")
                optimized_model, optimized_tokenizer = self._load_model_with_format_detection(optimized_model_path)
                console.print(f"[green]✓[/green] Optimized model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load optimized model: {e}")
                console.print(f"[bold red]Error loading optimized model:[/bold red] {str(e)}")
                raise ValueError(f"Failed to load optimized model: {e}")
                
            try:
                console.print(f"Loading baseline model: {baseline_model_path}")
                baseline_model, baseline_tokenizer = self._load_model_with_format_detection(baseline_model_path)
                console.print(f"[green]✓[/green] Baseline model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load baseline model: {e}")
                console.print(f"[bold red]Error loading baseline model:[/bold red] {str(e)}")
                raise ValueError(f"Failed to load baseline model: {e}")
            
            # Add model metadata to results
            metadata = {
                "optimized_model": {
                    "path": optimized_model_path,
                    "type": type(optimized_model).__name__,
                    "device": str(getattr(optimized_model, "device", "unknown")),
                },
                "baseline_model": {
                    "path": baseline_model_path,
                    "type": type(baseline_model).__name__,
                    "device": str(getattr(baseline_model, "device", "unknown")),
                },
                "benchmark_config": {
                    "sequence_lengths": str(self.config.get("sequence_lengths", [128, 512, 1024])),
                    "batch_sizes": str(self.config.get("batch_sizes", [1, 4, 8])),
                    "num_iterations": str(self.config.get("num_iterations", 10)),
                    "benchmark_quality": str(self.config.get("benchmark_quality", True)),
                    "benchmark_memory": str(self.config.get("benchmark_memory", True)),
                }
            }
            
            console.print(f"[bold blue]Running benchmarks...[/bold blue]")
                                                                                                                                                                                    
            # Run benchmarks                                                                                                                                                              
            benchmark_results = self._run_benchmarks(                                                                                                                                     
                optimized_model, optimized_tokenizer,                                                                                                                                     
                baseline_model, baseline_tokenizer                                                                                                                                        
            )
            
            # Add metadata to results
            benchmark_results["metadata"] = metadata
                                                                                                                                                                                    
            # Generate reports                                                                                                                                                            
            console.print(f"[bold blue]Generating reports...[/bold blue]")
            report_paths = self._generate_reports(benchmark_results, output_dir)                                                                                                          
                                                                                                                                                                                    
            logger.info(f"Benchmark reports saved to {output_dir}")
            console.print(f"[bold green]Benchmark reports saved to:[/bold green] {output_dir}")
                                                                                                                                                                                    
            # Return results                                                                                                                                                              
            return {                                                                                                                                                                      
                "model_state": model_state,  # Pass through unchanged                                                                                                                     
                "metrics": benchmark_results,                                                                                                                                             
                "artifacts": report_paths                                                                                                                                                 
            }
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            console.print(f"[bold red]Benchmarking failed:[/bold red] {str(e)}")
            
            # Create minimal output directory for error report
            os.makedirs(os.path.join(self.config.get("output_dir", "./outputs"), "benchmarks"), exist_ok=True)
            
            # Return error information
            return {
                "model_state": model_state,
                "metrics": {"error": str(e)},
                "artifacts": {}
            }
                                                                                                                                                                                    
    def _run_benchmarks(self, optimized_model, optimized_tokenizer, baseline_model, baseline_tokenizer):                                                                              
        """                                                                                                                                                                           
        Run comprehensive benchmarks comparing optimized and baseline models.                                                                                                         
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            optimized_model: The optimized model                                                                                                                                      
            optimized_tokenizer: Tokenizer for the optimized model                                                                                                                    
            baseline_model: The baseline model                                                                                                                                        
            baseline_tokenizer: Tokenizer for the baseline model                                                                                                                      
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of benchmark results                                                                                                                                           
        """                                                                                                                                                                           
        results = {                                                                                                                                                                   
            "latency": {},                                                                                                                                                            
            "memory": {},                                                                                                                                                             
            "throughput": {},                                                                                                                                                         
            "quality": {},                                                                                                                                                            
        }                                                                                                                                                                             
                                                                                                                                                                                    
        # Benchmark latency and throughput
        console.print("[bold]Running performance benchmarks...[/bold]")
        try:
            results["latency"], results["throughput"] = self._benchmark_performance(                                                                                                      
                optimized_model, baseline_model                                                                                                                                           
            )
            console.print("[green]✓[/green] Performance benchmarks completed")
        except Exception as e:
            logger.error(f"Performance benchmarking failed: {e}")
            console.print(f"[bold red]Performance benchmarking failed:[/bold red] {str(e)}")
            results["latency"] = {"error": str(e)}
            results["throughput"] = {"error": str(e)}
                                                                                                                                                                                    
        # Benchmark memory usage (if enabled)
        if self.config.get("benchmark_memory", True):
            console.print("[bold]Running memory benchmarks...[/bold]")
            try:
                results["memory"] = self._benchmark_memory(                                                                                                                   
                    optimized_model, baseline_model                                                                                                                           
                )
                console.print("[green]✓[/green] Memory benchmarks completed")
            except Exception as e:
                logger.error(f"Memory benchmarking failed: {e}")
                console.print(f"[bold red]Memory benchmarking failed:[/bold red] {str(e)}")
                results["memory"] = {"error": str(e)}
        else:
            console.print("[yellow]Skipping memory benchmarks (disabled in config)[/yellow]")
            results["memory"] = {"status": "skipped"}
                                                                                                                                                                                    
        # Benchmark quality (if enabled)                                                                                                                                              
        if self.config.get("benchmark_quality", True):
            console.print("[bold]Running quality benchmarks...[/bold]")
            try:
                results["quality"] = self._benchmark_quality(                                                                                                                             
                    optimized_model, optimized_tokenizer,                                                                                                                                 
                    baseline_model, baseline_tokenizer                                                                                                                                    
                )
                console.print("[green]✓[/green] Quality benchmarks completed")
            except Exception as e:
                logger.error(f"Quality benchmarking failed: {e}")
                console.print(f"[bold red]Quality benchmarking failed:[/bold red] {str(e)}")
                results["quality"] = {"error": str(e)}
        else:
            console.print("[yellow]Skipping quality benchmarks (disabled in config)[/yellow]")
            results["quality"] = {"status": "skipped"}
                                                                                                                                                                                    
        return results                                                                                                                                                                
                                                                                                                                                                                    
    def _benchmark_performance(self, optimized_model, baseline_model):                                                                                                                
        """                                                                                                                                                                           
        Benchmark model performance (latency and throughput).                                                                                                                         
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            optimized_model: The optimized model                                                                                                                                      
            baseline_model: The baseline model                                                                                                                                        
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Tuple of (latency_results, throughput_results)                                                                                                                            
        """                                                                                                                                                                           
        # Prepare benchmark parameters                                                                                                                                                
        sequence_lengths = self.config.get("sequence_lengths", [128, 512, 1024])                                                                                                      
        batch_sizes = self.config.get("batch_sizes", [1, 4, 8])                                                                                                                       
        num_iterations = self.config.get("num_iterations", 10)                                                                                                                        
                                                                                                                                                                                    
        latency_results = {}                                                                                                                                                          
        throughput_results = {}                                                                                                                                                       
                                                                                                                                                                                    
        # Set models to evaluation mode                                                                                                                                               
        optimized_model.eval()                                                                                                                                                        
        baseline_model.eval()                                                                                                                                                         
                                                                                                                                                                                    
        # Run benchmarks for each configuration                                                                                                                                       
        for seq_len in sequence_lengths:                                                                                                                                              
            for batch_size in batch_sizes:                                                                                                                                            
                # Skip large batch sizes for long sequences to avoid OOM                                                                                                              
                if seq_len * batch_size > 8192:                                                                                                                                       
                    continue                                                                                                                                                          
                                                                                                                                                                                    
                # Create random input                                                                                                                                                 
                input_ids = torch.randint(                                                                                                                                            
                    100, 1000,                                                                                                                                                        
                    (batch_size, seq_len),                                                                                                                                            
                    device=optimized_model.device                                                                                                                                     
                )                                                                                                                                                                     
                                                                                                                                                                                    
                # Benchmark optimized model                                                                                                                                           
                opt_latency, opt_throughput = self._measure_model_performance(                                                                                                        
                    optimized_model, input_ids, num_iterations                                                                                                                        
                )                                                                                                                                                                     
                                                                                                                                                                                    
                # Benchmark baseline model                                                                                                                                            
                base_latency, base_throughput = self._measure_model_performance(                                                                                                      
                    baseline_model, input_ids, num_iterations                                                                                                                         
                )                                                                                                                                                                     
                                                                                                                                                                                    
                # Store results                                                                                                                                                       
                config_key = f"b{batch_size}_s{seq_len}"                                                                                                                              
                latency_results[config_key] = {                                                                                                                                       
                    "optimized": opt_latency,                                                                                                                                         
                    "baseline": base_latency,                                                                                                                                         
                    "speedup": base_latency / opt_latency if opt_latency > 0 else 0,                                                                                                  
                }                                                                                                                                                                     
                                                                                                                                                                                    
                throughput_results[config_key] = {                                                                                                                                    
                    "optimized": opt_throughput,                                                                                                                                      
                    "baseline": base_throughput,                                                                                                                                      
                    "improvement": opt_throughput / base_throughput if base_throughput > 0 else 0,                                                                                    
                }                                                                                                                                                                     
                                                                                                                                                                                    
        return latency_results, throughput_results                                                                                                                                    
                                                                                                                                                                                    
    def _load_model_with_format_detection(self, model_path):
        """
        Load a model with automatic format detection.
        
        Supports:
        - Hugging Face models
        - GGUF models
        - ONNX models
        
        Args:
            model_path: Path to the model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if path is a GGUF file or contains GGUF in the path
        if model_path.endswith('.gguf') or '.gguf' in model_path:
            logger.info(f"Loading GGUF model: {model_path}")
            return load_gguf_model(model_path)
            
        # Check if path is an ONNX model directory
        elif os.path.exists(os.path.join(model_path, 'model.onnx')) or model_path.endswith('.onnx'):
            from optimum.onnxruntime import ORTModelForCausalLM
            try:
                logger.info(f"Loading ONNX model: {model_path}")
                model = ORTModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}")
                # Fall back to standard loading
        
        # Default: Load as Hugging Face model
        logger.info(f"Loading Hugging Face model: {model_path}")
        return load_model_and_tokenizer(model_path)
    
    def _measure_model_performance(self, model, input_ids, num_iterations):                                                                                                           
        """                                                                                                                                                                           
        Measure model latency and throughput.                                                                                                                                         
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to benchmark                                                                                                                                             
            input_ids: Input tensor                                                                                                                                                   
            num_iterations: Number of iterations to run                                                                                                                               
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Tuple of (latency_ms, throughput_samples_per_sec)                                                                                                                         
        """                                                                                                                                                                           
        batch_size = input_ids.size(0)                                                                                                                                                
        
        try:
            # Warm-up - with error handling
            for i in range(3):
                try:
                    with torch.no_grad():
                        _ = model(input_ids)
                except Exception as e:
                    logger.warning(f"Warm-up iteration {i} failed: {e}")
                    # Continue with benchmarking despite warm-up failure
                    break
                                                                                                                                                                                    
            # Measure performance - collect individual latencies to avoid duplication                                                                                                                                                         
            latencies = []
            
            # Track successful iterations
            successful_iterations = 0
                                                                                                                                                                                    
            for i in range(num_iterations):
                try:
                    # Synchronize before each iteration to get accurate per-iteration timing
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    iter_start = time.time()
                    
                    with torch.no_grad():                                                                                                                                                     
                        _ = model(input_ids)
                        
                    # Synchronize after iteration
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    iter_end = time.time()
                    
                    # Record individual latency for this iteration
                    iter_latency = (iter_end - iter_start) * 1000  # convert to ms
                    latencies.append(iter_latency)
                    
                    successful_iterations += 1
                    
                except Exception as e:
                    logger.error(f"Benchmark iteration {i} failed: {e}")
                    # Continue with remaining iterations
                                                                                                                                                                                    
            # Calculate metrics - handle case where all iterations failed                                                                                                                                                                           
            if successful_iterations == 0:
                logger.error("All benchmark iterations failed")
                return float('inf'), 0.0  # Indicate failure with infinite latency and zero throughput
                
            # Calculate average latency from individual measurements
            avg_latency_ms = sum(latencies) / len(latencies)
            
            # Calculate throughput based on average latency
            throughput = (batch_size * 1000) / avg_latency_ms
            
            logger.info(f"Performance: {avg_latency_ms:.2f}ms latency, {throughput:.2f} samples/sec "
                       f"({successful_iterations}/{num_iterations} successful iterations)")
                                                                                                                                                                                    
            return avg_latency_ms, throughput
            
        except Exception as e:
            logger.error(f"Performance measurement failed: {e}")
            return float('inf'), 0.0  # Indicate failure with infinite latency and zero throughput
                                                                                                                                                                                    
    def _benchmark_memory(self, optimized_model, baseline_model):                                                                                                                     
        """                                                                                                                                                                           
        Benchmark model memory usage.                                                                                                                                                 
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            optimized_model: The optimized model                                                                                                                                      
            baseline_model: The baseline model                                                                                                                                        
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of memory usage results                                                                                                                                        
        """
        try:
            # Get model sizes with error handling
            try:
                opt_size = self._get_model_size(optimized_model)
                logger.info(f"Optimized model size: {opt_size:.2f} MB")
            except Exception as e:
                logger.error(f"Error measuring optimized model size: {e}")
                opt_size = 0
                
            try:
                base_size = self._get_model_size(baseline_model)
                logger.info(f"Baseline model size: {base_size:.2f} MB")
            except Exception as e:
                logger.error(f"Error measuring baseline model size: {e}")
                base_size = 0
                                                                                                                                                                                    
            # Measure GPU memory if available                                                                                                                                             
            opt_gpu_memory = 0                                                                                                                                                            
            base_gpu_memory = 0                                                                                                                                                           
                                                                                                                                                                                    
            if torch.cuda.is_available():
                try:
                    # Measure optimized model GPU memory                                                                                                                                      
                    torch.cuda.reset_peak_memory_stats()                                                                                                                                      
                    torch.cuda.empty_cache()                                                                                                                                                  
                                                                                                                                                                                    
                    # Run a forward pass to allocate memory                                                                                                                                   
                    input_ids = torch.randint(                                                                                                                                                
                        100, 1000,                                                                                                                                                            
                        (1, 512),                                                                                                                                                             
                        device=optimized_model.device                                                                                                                                         
                    )                                                                                                                                                                         
                    with torch.no_grad():                                                                                                                                                     
                        _ = optimized_model(input_ids)                                                                                                                                        
                                                                                                                                                                                    
                    torch.cuda.synchronize()                                                                                                                                                  
                    opt_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                    logger.info(f"Optimized model GPU memory: {opt_gpu_memory:.2f} MB")
                except Exception as e:
                    logger.error(f"Error measuring optimized model GPU memory: {e}")
                    opt_gpu_memory = 0
                
                try:
                    # Measure baseline model GPU memory                                                                                                                                       
                    torch.cuda.reset_peak_memory_stats()                                                                                                                                      
                    torch.cuda.empty_cache()                                                                                                                                                  
                                                                                                                                                                                    
                    # Run a forward pass to allocate memory                                                                                                                                   
                    with torch.no_grad():                                                                                                                                                     
                        _ = baseline_model(input_ids)                                                                                                                                         
                                                                                                                                                                                    
                    torch.cuda.synchronize()                                                                                                                                                  
                    base_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                    logger.info(f"Baseline model GPU memory: {base_gpu_memory:.2f} MB")
                except Exception as e:
                    logger.error(f"Error measuring baseline model GPU memory: {e}")
                    base_gpu_memory = 0
            
            # Calculate reductions safely
            if base_size > 0:
                size_reduction = (base_size - opt_size) / base_size
            else:
                size_reduction = 0 if opt_size == 0 else float('inf')
                
            if base_gpu_memory > 0:
                gpu_reduction = (base_gpu_memory - opt_gpu_memory) / base_gpu_memory
            else:
                gpu_reduction = 0 if opt_gpu_memory == 0 else float('inf')
                                                                                                                                                                                    
            return {                                                                                                                                                                      
                "model_size_mb": {                                                                                                                                                        
                    "optimized": opt_size,                                                                                                                                                
                    "baseline": base_size,                                                                                                                                                
                    "reduction": size_reduction,
                    "reduction_mb": base_size - opt_size,
                },                                                                                                                                                                        
                "gpu_memory_mb": {                                                                                                                                                        
                    "optimized": opt_gpu_memory,                                                                                                                                          
                    "baseline": base_gpu_memory,                                                                                                                                          
                    "reduction": gpu_reduction,
                    "reduction_mb": base_gpu_memory - opt_gpu_memory,
                }                                                                                                                                                                         
            }
            
        except Exception as e:
            logger.error(f"Memory benchmarking failed: {e}")
            return {
                "model_size_mb": {"error": str(e)},
                "gpu_memory_mb": {"error": str(e)}
            }
                                                                                                                                                                                    
    def _get_model_size(self, model):                                                                                                                                                 
        """Calculate the size of a model in MB."""
        try:
            # Check if model has parameters() method
            if not hasattr(model, 'parameters'):
                # Try to get size from model attributes
                if hasattr(model, 'model_size_mb'):
                    return model.model_size_mb
                elif hasattr(model, 'model') and hasattr(model.model, 'model_size_mb'):
                    return model.model.model_size_mb
                else:
                    logger.warning(f"Cannot determine size for model type: {type(model)}")
                    return 0
                    
            # Standard parameter counting for PyTorch models
            param_size = 0                                                                                                                                                                
            param_count = 0
            for param in model.parameters():
                param_count += 1                                                                                                                                              
                param_size += param.nelement() * param.element_size()
            
            logger.debug(f"Model has {param_count} parameter tensors")
                                                                                                                                                                                    
            buffer_size = 0
            buffer_count = 0                                                                                                                                                               
            for buffer in model.buffers():
                buffer_count += 1                                                                                                                                                
                buffer_size += buffer.nelement() * buffer.element_size()
                
            logger.debug(f"Model has {buffer_count} buffer tensors")
                                                                                                                                                                                    
            size_mb = (param_size + buffer_size) / 1024 / 1024
            
            # Special case for GGUF models which might not report correct size through parameters
            if size_mb < 1.0 and hasattr(model, 'model') and hasattr(model.model, 'model_path'):
                # Try to get file size for GGUF models
                try:
                    import os
                    if os.path.exists(model.model.model_path):
                        file_size_mb = os.path.getsize(model.model.model_path) / 1024 / 1024
                        logger.info(f"Using GGUF file size: {file_size_mb:.2f} MB")
                        return file_size_mb
                except Exception as e:
                    logger.warning(f"Failed to get GGUF file size: {e}")
                                                                                                                                                                                    
            return size_mb
            
        except Exception as e:
            logger.error(f"Error calculating model size: {e}")
            return 0
                                                                                                                                                                                    
    def _benchmark_quality(self, optimized_model, optimized_tokenizer, baseline_model, baseline_tokenizer):                                                                           
        """                                                                                                                                                                           
        Benchmark model quality on various tasks.                                                                                                                                     
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            optimized_model: The optimized model                                                                                                                                      
            optimized_tokenizer: Tokenizer for the optimized model                                                                                                                    
            baseline_model: The baseline model                                                                                                                                        
            baseline_tokenizer: Tokenizer for the baseline model                                                                                                                      
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of quality benchmark results                                                                                                                                   
        """                                                                                                                                                                           
        try:
            # Add diagnostic logging for quality benchmarking
            logger.info("=== STARTING QUALITY BENCHMARKING ===")
            logger.info(f"Optimized model type: {type(optimized_model).__name__}")
            logger.info(f"Baseline model type: {type(baseline_model).__name__}")
            
            # Create a custom evaluator with diagnostic logging
            from llm_optimizer.evaluation import ModelEvaluator
            
            # Check if we should use bundled datasets
            use_bundled = self.config.get("evaluation", {}).get("use_bundled_datasets", False)
            if use_bundled:
                # Import the bundled dataset utilities
                try:
                    from llm_optimizer.utils.data import get_bundled_dataset_path
                    logger.info("Using bundled datasets for evaluation")
                    
                    # Update config to use bundled datasets
                    if "evaluation" not in self.config:
                        self.config["evaluation"] = {}
                    
                    # Set dataset paths in config
                    dataset = self.config["evaluation"].get("dataset", "tiny_wikitext")
                    completion_dataset = self.config["evaluation"].get("completion_dataset", "tiny_lambada")
                    
                    # Get paths to bundled datasets
                    self.config["evaluation"]["dataset_path"] = get_bundled_dataset_path(dataset)
                    self.config["evaluation"]["completion_dataset_path"] = get_bundled_dataset_path(completion_dataset)
                    
                    logger.info(f"Using bundled dataset: {dataset} at {self.config['evaluation']['dataset_path']}")
                    logger.info(f"Using bundled completion dataset: {completion_dataset} at {self.config['evaluation']['completion_dataset_path']}")
                    
                except ImportError as e:
                    logger.warning(f"Failed to import bundled dataset utilities: {e}")
                    logger.warning("Falling back to online datasets")
                except Exception as e:
                    logger.warning(f"Error setting up bundled datasets: {e}")
                    logger.warning("Falling back to online datasets")
            
            # Try to use the debug evaluator if available, otherwise fall back to regular evaluator
            try:
                from llm_optimizer.evaluation_debug import DebugModelEvaluator
                logger.info("Using DebugModelEvaluator for enhanced diagnostics")
                evaluator = DebugModelEvaluator(self.config)
            except ImportError:
                logger.info("Debug evaluator not available, using standard ModelEvaluator")
                from llm_optimizer.evaluation import ModelEvaluator
                evaluator = ModelEvaluator(self.config)
                                                                                                                                                                                    
            # Save models temporarily if needed                                                                                                                                           
            temp_dir = os.path.join(self.config.get("output_dir", "./outputs"), "temp")                                                                                                   
            os.makedirs(temp_dir, exist_ok=True)                                                                                                                                          
                                                                                                                                                                                    
            opt_temp_dir = os.path.join(temp_dir, "optimized")                                                                                                                            
            base_temp_dir = os.path.join(temp_dir, "baseline")                                                                                                                            
                                                                                                                                                                                    
            os.makedirs(opt_temp_dir, exist_ok=True)                                                                                                                                      
            os.makedirs(base_temp_dir, exist_ok=True)                                                                                                                                     
            
            # Handle different model types for saving
            self._save_model_with_format_detection(optimized_model, optimized_tokenizer, opt_temp_dir)
            self._save_model_with_format_detection(baseline_model, baseline_tokenizer, base_temp_dir)
                                                                                                                                                                                    
            # Evaluate models - use separate try/except blocks to handle individual failures
            opt_metrics = {}
            base_metrics = {}
            
            try:
                logger.info("Evaluating optimized model...")
                opt_metrics = evaluator.evaluate(opt_temp_dir)
            except Exception as e:
                logger.error(f"Error evaluating optimized model: {e}")
                opt_metrics = {"error": str(e)}
                
            try:
                logger.info("Evaluating baseline model...")
                base_metrics = evaluator.evaluate(base_temp_dir)
            except Exception as e:
                logger.error(f"Error evaluating baseline model: {e}")
                base_metrics = {"error": str(e)}
                                                                                                                                                                                    
            # Compare results                                                                                                                                                             
            quality_results = {}                                                                                                                                                          
                                                                                                                                                                                    
            # Add error information if present
            if "error" in opt_metrics:
                quality_results["optimized_model_error"] = {"error": opt_metrics["error"]}
                
            if "error" in base_metrics:
                quality_results["baseline_model_error"] = {"error": base_metrics["error"]}
            
            # Process metrics that are present in both results
            for metric in opt_metrics:
                if metric == "error":
                    continue
                    
                if metric in base_metrics:
                    # Skip NaN values
                    if (isinstance(opt_metrics[metric], float) and np.isnan(opt_metrics[metric])) or \
                       (isinstance(base_metrics[metric], float) and np.isnan(base_metrics[metric])):
                        quality_results[metric] = {
                            "optimized": opt_metrics[metric],
                            "baseline": base_metrics[metric],
                            "status": "NaN values detected"
                        }
                        continue
                        
                    quality_results[metric] = {                                                                                                                                           
                        "optimized": opt_metrics[metric],                                                                                                                                 
                        "baseline": base_metrics[metric],                                                                                                                                 
                    }                                                                                                                                                                     
                                                                                                                                                                                    
                    # Calculate relative difference only for non-zero, non-NaN values                                                                                                                      
                    if isinstance(base_metrics[metric], (int, float)) and base_metrics[metric] != 0 and \
                       isinstance(opt_metrics[metric], (int, float)):
                        rel_diff = (opt_metrics[metric] - base_metrics[metric]) / base_metrics[metric]                                                                                    
                        quality_results[metric]["relative_diff"] = rel_diff                                                                                                               
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Quality benchmarking failed: {e}")
            return {"benchmark_error": str(e)}
                                                                                                                                                                                    
    def _generate_reports(self, benchmark_results, output_dir):                                                                                                                       
        """                                                                                                                                                                           
        Generate benchmark reports and visualizations.                                                                                                                                
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            benchmark_results: Benchmark results                                                                                                                                      
            output_dir: Output directory                                                                                                                                              
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of report file paths                                                                                                                                           
        """                                                                                                                                                                           
        os.makedirs(output_dir, exist_ok=True)      

        # serializable_results = OmegaConf.to_container(benchmark_results, resolve=True)                                                                                                                                   
                                                                                                                                                                                    
        # Save raw results                                                                                                                                                            
        results_path = os.path.join(output_dir, "benchmark_results.json")                                                                                                             
        with open(results_path, "w") as f:                                                                                                                                            
            json.dump(benchmark_results, f, indent=2)                                                                                                                                 
                                                                                                                                                                                    
        # Generate summary report                                                                                                                                                     
        summary_path = os.path.join(output_dir, "benchmark_summary.txt")                                                                                                              
        self._generate_summary_report(benchmark_results, summary_path)                                                                                                                
                                                                                                                                                                                    
        # Generate visualizations                                                                                                                                                     
        viz_paths = self._generate_visualizations(benchmark_results, output_dir)                                                                                                      
                                                                                                                                                                                    
        return {                                                                                                                                                                      
            "results_json": results_path,                                                                                                                                             
            "summary_txt": summary_path,                                                                                                                                              
            **viz_paths                                                                                                                                                               
        }                                                                                                                                                                             
                                                                                                                                                                                    
    def _generate_summary_report(self, results, output_path):                                                                                                                         
        """Generate a text summary report."""                                                                                                                                         
        with open(output_path, "w") as f:                                                                                                                                             
            f.write("# LLM Optimization Benchmark Summary\n\n")                                                                                                                       
                                                                                                                                                                                    
            # Memory summary                                                                                                                                                          
            f.write("## Memory Usage\n\n")                                                                                                                                            
            mem_results = results["memory"]                                                                                                                                           
                                                                                                                                                                                    
            f.write(f"Model Size:\n")                                                                                                                                                 
            f.write(f"  - Baseline: {mem_results['model_size_mb']['baseline']:.2f} MB\n")                                                                                             
            f.write(f"  - Optimized: {mem_results['model_size_mb']['optimized']:.2f} MB\n")                                                                                           
            f.write(f"  - Reduction: {mem_results['model_size_mb']['reduction']*100:.2f}%\n\n")                                                                                       
                                                                                                                                                                                    
            f.write(f"GPU Memory:\n")                                                                                                                                                 
            f.write(f"  - Baseline: {mem_results['gpu_memory_mb']['baseline']:.2f} MB\n")                                                                                             
            f.write(f"  - Optimized: {mem_results['gpu_memory_mb']['optimized']:.2f} MB\n")                                                                                           
            f.write(f"  - Reduction: {mem_results['gpu_memory_mb']['reduction']*100:.2f}%\n\n")                                                                                       
                                                                                                                                                                                    
            # Latency summary                                                                                                                                                         
            f.write("## Latency\n\n")                                                                                                                                                 
            f.write("| Configuration | Baseline (ms) | Optimized (ms) | Speedup |\n")                                                                                                 
            f.write("|--------------|--------------|---------------|--------|\n")                                                                                                     
                                                                                                                                                                                    
            for config, values in results["latency"].items():                                                                                                                         
                f.write(f"| {config} | {values['baseline']:.2f} | {values['optimized']:.2f} | {values['speedup']:.2f}x |\n")                                                          
                                                                                                                                                                                    
            f.write("\n")                                                                                                                                                             
                                                                                                                                                                                    
            # Throughput summary                                                                                                                                                      
            f.write("## Throughput (samples/sec)\n\n")                                                                                                                                
            f.write("| Configuration | Baseline | Optimized | Improvement |\n")                                                                                                       
            f.write("|--------------|----------|-----------|-------------|\n")                                                                                                        
                                                                                                                                                                                    
            for config, values in results["throughput"].items():                                                                                                                      
                f.write(f"| {config} | {values['baseline']:.2f} | {values['optimized']:.2f} | {values['improvement']:.2f}x |\n")                                                      
                                                                                                                                                                                    
            f.write("\n")                                                                                                                                                             
                                                                                                                                                                                    
            # Quality summary (if available)                                                                                                                                          
            if "quality" in results and results["quality"]:                                                                                                                           
                f.write("## Quality Metrics\n\n")                                                                                                                                     
                f.write("| Metric | Baseline | Optimized | Relative Difference |\n")                                                                                                  
                f.write("|--------|----------|-----------|---------------------|\n")                                                                                                  
                                                                                                                                                                                    
                for metric, values in results["quality"].items():
                    if metric in ["optimized_model_error", "baseline_model_error", "benchmark_error"]:
                        continue
                                                                                                                                                                                    
                    rel_diff = values.get("relative_diff", "N/A")                                                                                                                     
                    if isinstance(rel_diff, float):                                                                                                                                   
                        rel_diff = f"{rel_diff*100:+.2f}%"                                                                                                                            
                                                                                                                                                                                    
                    f.write(f"| {metric} | {values['baseline']} | {values['optimized']} | {rel_diff} |\n")
                
                # Add error information if present
                for error_key in ["benchmark_error", "optimized_model_error", "baseline_model_error"]:
                    if error_key in results["quality"]:
                        f.write(f"\n**Error in {error_key}**: {results['quality'][error_key]}\n")
                                                                                                                                                                                    
    def _generate_visualizations(self, results, output_dir):                                                                                                                          
        """Generate visualizations of benchmark results."""                                                                                                                           
        viz_paths = {}                                                                                                                                                                
                                                                                                                                                                                    
        # Set up plotting style                                                                                                                                                       
        plt.style.use("seaborn-v0_8-darkgrid")                                                                                                                                        
                                                                                                                                                                                    
        # Latency comparison                                                                                                                                                          
        latency_path = os.path.join(output_dir, "latency_comparison.png")                                                                                                             
        self._plot_comparison(                                                                                                                                                        
            results["latency"],                                                                                                                                                       
            "Latency (ms)",                                                                                                                                                           
            "Lower is better",                                                                                                                                                        
            latency_path                                                                                                                                                              
        )                                                                                                                                                                             
        viz_paths["latency_plot"] = latency_path                                                                                                                                      
                                                                                                                                                                                    
        # Throughput comparison                                                                                                                                                       
        throughput_path = os.path.join(output_dir, "throughput_comparison.png")                                                                                                       
        self._plot_comparison(                                                                                                                                                        
            results["throughput"],                                                                                                                                                    
            "Throughput (samples/sec)",                                                                                                                                               
            "Higher is better",                                                                                                                                                       
            throughput_path,                                                                                                                                                          
            higher_better=True                                                                                                                                                        
        )                                                                                                                                                                             
        viz_paths["throughput_plot"] = throughput_path                                                                                                                                
                                                                                                                                                                                    
        # Memory usage                                                                                                                                                                
        memory_path = os.path.join(output_dir, "memory_usage.png")                                                                                                                    
        self._plot_memory_usage(results["memory"], memory_path)                                                                                                                       
        viz_paths["memory_plot"] = memory_path                                                                                                                                        
                                                                                                                                                                                    
        # Quality comparison (if available)                                                                                                                                           
        if "quality" in results and results["quality"]:                                                                                                                               
            quality_path = os.path.join(output_dir, "quality_comparison.png")                                                                                                         
            self._plot_quality_comparison(results["quality"], quality_path)                                                                                                           
            viz_paths["quality_plot"] = quality_path                                                                                                                                  
                                                                                                                                                                                    
        return viz_paths                                                                                                                                                              
                                                                                                                                                                                    
    def _plot_comparison(self, data, ylabel, title, output_path, higher_better=False):                                                                                                
        """Plot comparison between optimized and baseline models."""                                                                                                                  
        # Prepare data                                                                                                                                                                
        configs = list(data.keys())                                                                                                                                                   
        baseline_values = [data[c]["baseline"] for c in configs]                                                                                                                      
        optimized_values = [data[c]["optimized"] for c in configs]                                                                                                                    
                                                                                                                                                                                    
        # Create figure                                                                                                                                                               
        fig, ax = plt.subplots(figsize=(10, 6))                                                                                                                                       
                                                                                                                                                                                    
        # Set width of bars                                                                                                                                                           
        bar_width = 0.35                                                                                                                                                              
        x = np.arange(len(configs))                                                                                                                                                   
                                                                                                                                                                                    
        # Create bars                                                                                                                                                                 
        ax.bar(x - bar_width/2, baseline_values, bar_width, label="Baseline", color="skyblue")                                                                                        
        ax.bar(x + bar_width/2, optimized_values, bar_width, label="Optimized", color="orange")                                                                                       
                                                                                                                                                                                    
        # Add labels and title                                                                                                                                                        
        ax.set_xlabel("Configuration")                                                                                                                                                
        ax.set_ylabel(ylabel)                                                                                                                                                         
        ax.set_title(title)                                                                                                                                                           
        ax.set_xticks(x)                                                                                                                                                              
        ax.set_xticklabels(configs, rotation=45)                                                                                                                                      
        ax.legend()                                                                                                                                                                   
                                                                                                                                                                                    
        # Add improvement percentages                                                                                                                                                 
        for i, config in enumerate(configs):                                                                                                                                          
            baseline = data[config]["baseline"]                                                                                                                                       
            optimized = data[config]["optimized"]                                                                                                                                     
                                                                                                                                                                                    
            if higher_better:                                                                                                                                                         
                pct = (optimized / baseline - 1) * 100 if baseline > 0 else 0                                                                                                         
                text = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"                                                                                                                   
                color = "green" if pct > 0 else "red"                                                                                                                                 
            else:                                                                                                                                                                     
                pct = (1 - optimized / baseline) * 100 if baseline > 0 else 0                                                                                                         
                text = f"-{pct:.1f}%" if pct > 0 else f"+{-pct:.1f}%"                                                                                                                 
                color = "green" if pct > 0 else "red"                                                                                                                                 
                                                                                                                                                                                    
            ax.text(i + bar_width/2, optimized, text, ha="center", va="bottom", color=color)                                                                                          
                                                                                                                                                                                    
        # Adjust layout and save                                                                                                                                                      
        fig.tight_layout()                                                                                                                                                            
        plt.savefig(output_path)                                                                                                                                                      
        plt.close(fig)                                                                                                                                                                
                                                                                                                                                                                    
    def _plot_memory_usage(self, memory_data, output_path):                                                                                                                           
        """Plot memory usage comparison."""                                                                                                                                           
        # Prepare data                                                                                                                                                                
        categories = ["Model Size", "GPU Memory"]                                                                                                                                     
        baseline_values = [                                                                                                                                                           
            memory_data["model_size_mb"]["baseline"],                                                                                                                                 
            memory_data["gpu_memory_mb"]["baseline"]                                                                                                                                  
        ]                                                                                                                                                                             
        optimized_values = [                                                                                                                                                          
            memory_data["model_size_mb"]["optimized"],                                                                                                                                
            memory_data["gpu_memory_mb"]["optimized"]                                                                                                                                 
        ]                                                                                                                                                                             
                                                                                                                                                                                    
        # Create figure                                                                                                                                                               
        fig, ax = plt.subplots(figsize=(8, 6))                                                                                                                                        
                                                                                                                                                                                    
        # Set width of bars                                                                                                                                                           
        bar_width = 0.35                                                                                                                                                              
        x = np.arange(len(categories))                                                                                                                                                
                                                                                                                                                                                    
        # Create bars                                                                                                                                                                 
        ax.bar(x - bar_width/2, baseline_values, bar_width, label="Baseline", color="skyblue")                                                                                        
        ax.bar(x + bar_width/2, optimized_values, bar_width, label="Optimized", color="orange")                                                                                       
                                                                                                                                                                                    
        # Add labels and title                                                                                                                                                        
        ax.set_xlabel("Memory Type")                                                                                                                                                  
        ax.set_ylabel("Memory Usage (MB)")                                                                                                                                            
        ax.set_title("Memory Usage Comparison")                                                                                                                                       
        ax.set_xticks(x)                                                                                                                                                              
        ax.set_xticklabels(categories)                                                                                                                                                
        ax.legend()                                                                                                                                                                   
                                                                                                                                                                                    
        # Add reduction percentages                                                                                                                                                   
        for i, category in enumerate(categories):                                                                                                                                     
            baseline = baseline_values[i]                                                                                                                                             
            optimized = optimized_values[i]                                                                                                                                           
                                                                                                                                                                                    
            pct = (1 - optimized / baseline) * 100 if baseline > 0 else 0                                                                                                             
            text = f"-{pct:.1f}%"                                                                                                                                                     
                                                                                                                                                                                    
            ax.text(i + bar_width/2, optimized, text, ha="center", va="bottom", color="green")                                                                                        
                                                                                                                                                                                    
        # Adjust layout and save                                                                                                                                                      
        fig.tight_layout()                                                                                                                                                            
        plt.savefig(output_path)                                                                                                                                                      
        plt.close(fig)                                                                                                                                                                
                                                                                                                                                                                    
    def _save_model_with_format_detection(self, model, tokenizer, output_dir):
        """
        Save a model with format detection.
        
        Handles different model types:
        - Hugging Face models
        - GGUF models
        - ONNX models
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_dir: Directory to save to
        """
        # Check if model is a GGUF model wrapper
        if hasattr(model, 'name_or_path') and getattr(model, 'name_or_path', '').endswith('.gguf'):
            # For GGUF models, we just need to copy the file
            import shutil
            gguf_path = model.name_or_path
            shutil.copy(gguf_path, os.path.join(output_dir, os.path.basename(gguf_path)))
            
            # Save tokenizer if possible
            if hasattr(tokenizer, 'save_pretrained'):
                try:
                    tokenizer.save_pretrained(output_dir)
                except Exception as e:
                    logger.warning(f"Could not save GGUF tokenizer: {e}")
            return
            
        # Check if model is an ONNX model
        if hasattr(model, 'model_path') and model.model_path.endswith('.onnx'):
            # For ONNX models, copy the model file
            import shutil
            shutil.copy(model.model_path, os.path.join(output_dir, 'model.onnx'))
            
            # Save tokenizer if possible
            if hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(output_dir)
            return
        
        # Default: Save as Hugging Face model
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            logger.warning(f"Error saving model: {e}")
            # Fallback: try to copy the model files
            if hasattr(model, 'name_or_path') and os.path.exists(model.name_or_path):
                import shutil
                if os.path.isdir(model.name_or_path):
                    for item in os.listdir(model.name_or_path):
                        s = os.path.join(model.name_or_path, item)
                        d = os.path.join(output_dir, item)
                        if os.path.isdir(s):
                            shutil.copytree(s, d, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s, d)
                else:
                    shutil.copy2(model.name_or_path, output_dir)
    
    def _plot_quality_comparison(self, quality_data, output_path):                                                                                                                    
        """Plot quality metrics comparison."""                                                                                                                                        
        # Filter metrics that have numeric values                                                                                                                                     
        numeric_metrics = {}                                                                                                                                                          
        for metric, values in quality_data.items():                                                                                                                                   
            if (isinstance(values["baseline"], (int, float)) and                                                                                                                      
                isinstance(values["optimized"], (int, float))):                                                                                                                       
                numeric_metrics[metric] = values                                                                                                                                      
                                                                                                                                                                                    
        if not numeric_metrics:                                                                                                                                                       
            return                                                                                                                                                                    
                                                                                                                                                                                    
        # Prepare data                                                                                                                                                                
        metrics = list(numeric_metrics.keys())                                                                                                                                        
        baseline_values = [numeric_metrics[m]["baseline"] for m in metrics]                                                                                                           
        optimized_values = [numeric_metrics[m]["optimized"] for m in metrics]                                                                                                         
                                                                                                                                                                                    
        # Create figure                                                                                                                                                               
        fig, ax = plt.subplots(figsize=(10, 6))                                                                                                                                       
                                                                                                                                                                                    
        # Set width of bars                                                                                                                                                           
        bar_width = 0.35                                                                                                                                                              
        x = np.arange(len(metrics))                                                                                                                                                   
                                                                                                                                                                                    
        # Create bars                                                                                                                                                                 
        ax.bar(x - bar_width/2, baseline_values, bar_width, label="Baseline", color="skyblue")                                                                                        
        ax.bar(x + bar_width/2, optimized_values, bar_width, label="Optimized", color="orange")                                                                                       
                                                                                                                                                                                    
        # Add labels and title                                                                                                                                                        
        ax.set_xlabel("Metric")                                                                                                                                                       
        ax.set_ylabel("Value")                                                                                                                                                        
        ax.set_title("Quality Metrics Comparison")                                                                                                                                    
        ax.set_xticks(x)                                                                                                                                                              
        ax.set_xticklabels(metrics, rotation=45)                                                                                                                                      
        ax.legend()                                                                                                                                                                   
                                                                                                                                                                                    
        # Add relative difference                                                                                                                                                     
        for i, metric in enumerate(metrics):                                                                                                                                          
            baseline = numeric_metrics[metric]["baseline"]                                                                                                                            
            optimized = numeric_metrics[metric]["optimized"]                                                                                                                          
                                                                                                                                                                                    
            if "relative_diff" in numeric_metrics[metric]:                                                                                                                            
                rel_diff = numeric_metrics[metric]["relative_diff"] * 100                                                                                                             
                text = f"{rel_diff:+.1f}%"                                                                                                                                            
                color = "green" if rel_diff >= 0 else "red"                                                                                                                           
                                                                                                                                                                                    
                ax.text(i + bar_width/2, optimized, text, ha="center", va="bottom", color=color)                                                                                      
                                                                                                                                                                                    
        # Adjust layout and save                                                                                                                                                      
        fig.tight_layout()                                                                                                                                                            
        plt.savefig(output_path)                                                                                                                                                      
        plt.close(fig) 
