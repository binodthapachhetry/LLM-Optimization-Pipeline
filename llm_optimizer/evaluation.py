"""                                                                                                                                                                                   
Evaluation module for LLM optimization.                                                                                                                                               
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import time                                                                                                                                                                           
import json                                                                                                                                                                           
import logging                                                                                                                                                                        
from typing import Dict, Any, List, Optional, Union                                                                                                                                   
                                                                                                                                                                                    
import torch                                                                                                                                                                          
import numpy as np                                                                                                                                                                    
from transformers import AutoModelForCausalLM, AutoTokenizer                                                                                                                          
from datasets import load_dataset                                                                                                                                       
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage, Evaluator                                                                                                                           
from llm_optimizer.utils.model import load_model_and_tokenizer                                                                                                                        
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
class EvaluationStage(OptimizationStage):                                                                                                                                             
    """                                                                                                                                                                               
    Evaluation stage for LLM optimization.                                                                                                                                            
                                                                                                                                                                                    
    Evaluates model performance using various metrics:                                                                                                                                
    - Perplexity                                                                                                                                                                      
    - Accuracy                                                                                                                                                                        
    - Latency                                                                                                                                                                         
    - Memory usage                                                                                                                                                                    
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                     
        """                                                                                                                                                                           
        Run the evaluation stage.                                                                                                                                                     
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_state: Current state of the model                                                                                                                                   
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Evaluation metrics                                                                                                                                                        
        """                                                                                                                                                                           
        self.validate_input(model_state)                                                                                                                                              
                                                                                                                                                                                    
        # Extract configuration                                                                                                                                                       
        model_path = model_state["model_path"]                                                                                                                                        
        output_dir = os.path.join(                                                                                                                                                    
            self.config.get("output_dir", "./outputs"),                                                                                                                               
            "evaluation"                                                                                                                                                              
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Create evaluator                                                                                                                                                            
        evaluator = ModelEvaluator(self.config)                                                                                                                                       
                                                                                                                                                                                    
        # Run evaluation                                                                                                                                                              
        logger.info(f"Evaluating model {model_path}")                                                                                                                                 
        metrics = evaluator.evaluate(model_path)                                                                                                                                      
                                                                                                                                                                                    
        # Save evaluation results                                                                                                                                                     
        os.makedirs(output_dir, exist_ok=True)                                                                                                                                        
        results_path = os.path.join(output_dir, "evaluation_results.json")                                                                                                            
                                                                                                                                                                                    
        with open(results_path, "w") as f:                                                                                                                                            
            json.dump(metrics, f, indent=2)                                                                                                                                           
                                                                                                                                                                                    
        logger.info(f"Evaluation results saved to {results_path}")                                                                                                                    
                                                                                                                                                                                    
        # Return metrics                                                                                                                                                              
        return {                                                                                                                                                                      
            "model_state": model_state,  # Pass through unchanged                                                                                                                     
            "metrics": metrics,                                                                                                                                                       
            "artifacts": {                                                                                                                                                            
                "results_path": results_path,                                                                                                                                         
            }                                                                                                                                                                         
        }                                                                                                                                                                             
                                                                                                                                                                                    
                                                                                                                                                                                    
class ModelEvaluator(Evaluator):                                                                                                                                                      
    """                                                                                                                                                                               
    Comprehensive model evaluator.                                                                                                                                                    
                                                                                                                                                                                    
    Evaluates models on various dimensions:                                                                                                                                           
    - Intrinsic metrics (perplexity)                                                                                                                                                  
    - Task performance (accuracy, F1, etc.)                                                                                                                                           
    - Efficiency (latency, throughput, memory)                                                                                                                                        
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def evaluate(self, model_path: str) -> Dict[str, Any]:                                                                                                                            
        """                                                                                                                                                                           
        Evaluate a model comprehensively.                                                                                                                                             
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_path: Path to the model                                                                                                                                             
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of evaluation metrics                                                                                                                                          
        """                                                                                                                                                                           
        metrics = {}                                                                                                                                                                  
                                                                                                                                                                                    
        # Load model and tokenizer                                                                                                                                                    
        model, tokenizer = load_model_and_tokenizer(model_path)                                                                                                                       
                                                                                                                                                                                    
        # Evaluate intrinsic metrics                                                                                                                                                  
        if self.config.get("evaluate_perplexity", True):                                                                                                                              
            metrics.update(self._evaluate_perplexity(model, tokenizer))                                                                                                               
                                                                                                                                                                                    
        # Evaluate task performance                                                                                                                                                   
        if self.config.get("evaluate_tasks", True):                                                                                                                                   
            metrics.update(self._evaluate_tasks(model, tokenizer))                                                                                                                    
                                                                                                                                                                                    
        # Evaluate efficiency                                                                                                                                                         
        if self.config.get("evaluate_efficiency", True):                                                                                                                              
            metrics.update(self._evaluate_efficiency(model, tokenizer))                                                                                                               
                                                                                                                                                                                    
        # Evaluate memory usage                                                                                                                                                       
        if self.config.get("evaluate_memory", True):                                                                                                                                  
            metrics.update(self._evaluate_memory(model))                                                                                                                              
                                                                                                                                                                                    
        return metrics                                                                                                                                                                
                                                                                                                                                                                    
    def _evaluate_perplexity(self, model, tokenizer) -> Dict[str, float]:                                                                                                             
        """                                                                                                                                                                           
        Evaluate model perplexity on a dataset.                                                                                                                                       
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to evaluate                                                                                                                                              
            tokenizer: The tokenizer                                                                                                                                                  
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of perplexity metrics                                                                                                                                          
        """                                                                                                                                                                           
        logger.info("Evaluating perplexity")                                                                                                                                          
                                                                                                                                                                                    
        # Load dataset                                                                                                                                                                
        dataset_name = self.config.get("perplexity_dataset", "wikitext")                                                                                                              
        dataset_config = self.config.get("perplexity_dataset_config", "wikitext-2-raw-v1")                                                                                            
                                                                                                                                                                                    
        try:                                                                                                                                                                          
            dataset = load_dataset(dataset_name, dataset_config, split="validation")                                                                                                  
        except Exception as e:                                                                                                                                                        
            logger.error(f"Failed to load dataset {dataset_name}/{dataset_config}: {e}")                                                                                              
            return {"perplexity": float("nan")}   

        # Get max sequence length from config                                                                                                                                             
        max_length = self.config.get("evaluation", {}).get("max_sequence_length", 1024)                                                                                                   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        # Tokenize dataset                                                                                                                                                            
        encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", max_length=max_length, truncation=True)                                                                                                      
                                                                                                                                                                                    
        # Calculate perplexity                                                                                                                                                        
        stride = self.config.get("stride", 512)                                                                                                                                       
                                                                                                                                                                                    
        lls = []                                                                                                                                                                      
        for i in range(0, encodings.input_ids.size(1), stride):                                                                                                                       
            begin_loc = max(i + stride - max_length, 0)                                                                                                                               
            end_loc = min(i + stride, encodings.input_ids.size(1))                                                                                                                    
            trg_len = end_loc - i                                                                                                                                                     
                                                                                                                                                                                    
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)                                                                                                    
            target_ids = input_ids.clone()                                                                                                                                            
            target_ids[:, :-trg_len] = -100                                                                                                                                           
                                                                                                                                                                                    
            with torch.no_grad():                                                                                                                                                     
                outputs = model(input_ids, labels=target_ids)                                                                                                                         
                                                                                                                                                                                    
                # Get loss                                                                                                                                                            
                log_likelihood = outputs.loss * trg_len                                                                                                                               
                                                                                                                                                                                    
            lls.append(log_likelihood)                                                                                                                                                
                                                                                                                                                                                    
        # Calculate perplexity                                                                                                                                                        
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)                                                                                                                             
                                                                                                                                                                                    
        return {"perplexity": ppl.item()}                                                                                                                                             
                                                                                                                                                                                    
    def _evaluate_tasks(self, model, tokenizer) -> Dict[str, float]:                                                                                                                  
        """                                                                                                                                                                           
        Evaluate model on downstream tasks.                                                                                                                                           
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to evaluate                                                                                                                                              
            tokenizer: The tokenizer                                                                                                                                                  
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of task performance metrics                                                                                                                                    
        """                                                                                                                                                                           
        logger.info("Evaluating task performance")                                                                                                                                    
                                                                                                                                                                                    
        # Get tasks to evaluate                                                                                                                                                       
        tasks = self.config.get("tasks", ["lambada"])                                                                                                                                 
                                                                                                                                                                                    
        metrics = {}                                                                                                                                                                  
                                                                                                                                                                                    
        # Evaluate each task                                                                                                                                                          
        for task in tasks:                                                                                                                                                            
            if task == "lambada":                                                                                                                                                     
                metrics.update(self._evaluate_lambada(model, tokenizer))                                                                                                              
            elif task == "hellaswag":                                                                                                                                                 
                metrics.update(self._evaluate_hellaswag(model, tokenizer))                                                                                                            
            else:                                                                                                                                                                     
                logger.warning(f"Unknown task: {task}")                                                                                                                               
                                                                                                                                                                                    
        return metrics                                                                                                                                                                
                                                                                                                                                                                    
    def _evaluate_lambada(self, model, tokenizer) -> Dict[str, float]:                                                                                                                
        """                                                                                                                                                                           
        Evaluate on LAMBADA dataset (last word prediction).                                                                                                                           
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to evaluate                                                                                                                                              
            tokenizer: The tokenizer                                                                                                                                                  
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of LAMBADA metrics                                                                                                                                             
        """                                                                                                                                                                           
        try:                                                                                                                                                                          
            # Load dataset                                                                                                                                                            
            dataset = load_dataset("lambada", split="validation")                                                                                                                     
                                                                                                                                                                                    
            # Evaluate on a subset for efficiency                                                                                                                                     
            max_samples = self.config.get("max_samples", 100)                                                                                                                         
            if len(dataset) > max_samples:                                                                                                                                            
                dataset = dataset.select(range(max_samples))                                                                                                                          
                                                                                                                                                                                    
            correct = 0                                                                                                                                                               
            total = 0   
                                                                                                                                                                                    
            for example in dataset:                                                                                                                                                   
                text = example["text"]                                                                                                                                                
                                                                                                                                                                                    
                # Get the last word                                                                                                                                                   
                last_word = text.split()[-1]                                                                                                                                          
                                                                                                                                                                                    
                # Get the context (everything except the last word)                                                                                                                   
                context = text[:-len(last_word)].strip()                                                                                                                              
                                                                                                                                                                                    
                # Tokenize                                                                                                                                                            
                input_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)                                                                                     
                                                                                                                                                                                    
                # Generate                                                                                                                                                            
                with torch.no_grad():                                                                                                                                                 
                    output = model.generate(                                                                                                                                          
                        input_ids,                                                                                                                                                    
                        max_new_tokens=5,                                                                                                                                             
                        num_return_sequences=1,                                                                                                                                       
                        pad_token_id=tokenizer.eos_token_id                                                                                                                           
                    )                                                                                                                                                                 
                                                                                                                                                                                    
                # Decode                                                                                                                                                              
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)                                                                                                
                                                                                                                                                                                    
                # Check if the generated text contains the last word                                                                                                                  
                if last_word in generated_text[len(context):]:                                                                                                                        
                    correct += 1                                                                                                                                                      
                                                                                                                                                                                    
                total += 1                                                                                                                                                            
                                                                                                                                                                                    
            accuracy = correct / total if total > 0 else 0                                                                                                                            
                                                                                                                                                                                    
            return {"lambada_accuracy": accuracy}                                                                                                                                     
                                                                                                                                                                                    
        except Exception as e:                                                                                                                                                        
            logger.error(f"Failed to evaluate on LAMBADA: {e}")                                                                                                                       
            return {"lambada_accuracy": float("nan")}                                                                                                                                 
                                                                                                                                                                                    
    def _evaluate_hellaswag(self, model, tokenizer) -> Dict[str, float]:                                                                                                              
        """                                                                                                                                                                           
        Evaluate on HellaSwag dataset (commonsense reasoning).                                                                                                                        
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to evaluate                                                                                                                                              
            tokenizer: The tokenizer                                                                                                                                                  
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of HellaSwag metrics                                                                                                                                           
        """                                                                                                                                                                           
        try:                                                                                                                                                                          
            # Load dataset                                                                                                                                                            
            dataset = load_dataset("hellaswag", "default", split="validation")                                                                                                        
                                                                                                                                                                                    
            # Evaluate on a subset for efficiency                                                                                                                                     
            max_samples = self.config.get("max_samples", 100)                                                                                                                         
            if len(dataset) > max_samples:                                                                                                                                            
                dataset = dataset.select(range(max_samples))                                                                                                                          
                                                                                                                                                                                    
            correct = 0                                                                                                                                                               
            total = 0                                                                                                                                                                 
                                                                                                                                                                                    
            for example in dataset:                                                                                                                                                   
                context = example["ctx"]                                                                                                                                              
                endings = example["endings"]                                                                                                                                          
                label = int(example["label"])                                                                                                                                         
                                                                                                                                                                                    
                # Score each ending                                                                                                                                                   
                scores = []                                                                                                                                                           
                                                                                                                                                                                    
                for ending in endings:                                                                                                                                                
                    # Combine context and ending                                                                                                                                      
                    full_text = context + " " + ending                                                                                                                                
                                                                                                                                                                                    
                    # Tokenize                                                                                                                                                        
                    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)                                                                                               
                                                                                                                                                                                    
                    # Calculate likelihood                                                                                                                                            
                    with torch.no_grad():                                                                                                                                             
                        outputs = model(**inputs, labels=inputs.input_ids)                                                                                                            
                                                                                                                                                                                    
                    # Use loss as negative log likelihood                                                                                                                             
                    score = -outputs.loss.item()                                                                                                                                      
                    scores.append(score)                                                                                                                                              
                                                                                                                                                                                    
                # Get prediction (highest score)                                                                                                                                      
                prediction = np.argmax(scores)                                                                                                                                        
                                                                                                                                                                                    
                if prediction == label:                                                                                                                                               
                    correct += 1                                                                                                                                                      
                                                                                                                                                                                    
                total += 1                                                                                                                                                            
                                                                                                                                                                                    
            accuracy = correct / total if total > 0 else 0                                                                                                                            
                                                                                                                                                                                    
            return {"hellaswag_accuracy": accuracy}                                                                                                                                   
                                                                                                                                                                                    
        except Exception as e:                                                                                                                                                        
            logger.error(f"Failed to evaluate on HellaSwag: {e}")                                                                                                                     
            return {"hellaswag_accuracy": float("nan")}                                                                                                                               
                                                                                                                                                                                    
    def _evaluate_efficiency(self, model, tokenizer) -> Dict[str, float]:                                                                                                             
        """                                                                                                                                                                           
        Evaluate model efficiency (latency, throughput).                                                                                                                              
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to evaluate                                                                                                                                              
            tokenizer: The tokenizer                                                                                                                                                  
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of efficiency metrics                                                                                                                                          
        """                                                                                                                                                                           
        logger.info("Evaluating efficiency")                                                                                                                                          
                                                                                                                                                                                    
        # Prepare input sequences                                                                                                                                                     
        sequence_lengths = self.config.get("sequence_lengths", [128, 512, 1024])                                                                                                      
        batch_sizes = self.config.get("batch_sizes", [1, 4])                                                                                                                          
        num_iterations = self.config.get("num_iterations", 10)                                                                                                                        
                                                                                                                                                                                    
        metrics = {}                                                                                                                                                                  
                                                                                                                                                                                    
        # Measure inference latency and throughput                                                                                                                                    
        for seq_len in sequence_lengths:                                                                                                                                              
            for batch_size in batch_sizes:                                                                                                                                            
                # Skip large batch sizes for long sequences to avoid OOM                                                                                                              
                if seq_len * batch_size > 8192:                                                                                                                                       
                    continue                                                                                                                                                          
                                                                                                                                                                                    
                # Create random input                                                                                                                                                 
                input_ids = torch.randint(                                                                                                                                            
                    100, 1000,                                                                                                                                                        
                    (batch_size, seq_len),                                                                                                                                            
                    device=model.device                                                                                                                                               
                )                                                                                                                                                                     
                                                                                                                                                                                    
                # Warm-up                                                                                                                                                             
                for _ in range(3):                                                                                                                                                    
                    with torch.no_grad():                                                                                                                                             
                        _ = model(input_ids)                                                                                                                                          
                                                                                                                                                                                    
                # Measure latency                                                                                                                                                     
                start_time = time.time()                                                                                                                                              
                                                                                                                                                                                    
                for _ in range(num_iterations):                                                                                                                                       
                    with torch.no_grad():                                                                                                                                             
                        _ = model(input_ids)                                                                                                                                          
                                                                                                                                                                                    
                end_time = time.time()                                                                                                                                                
                                                                                                                                                                                    
                # Calculate metrics                                                                                                                                                   
                total_time = end_time - start_time                                                                                                                                    
                latency_ms = (total_time / num_iterations) * 1000                                                                                                                     
                throughput = (batch_size * num_iterations) / total_time                                                                                                               
                                                                                                                                                                                    
                # Store metrics                                                                                                                                                       
                metrics[f"latency_ms_b{batch_size}_s{seq_len}"] = latency_ms                                                                                                          
                metrics[f"throughput_samples_per_sec_b{batch_size}_s{seq_len}"] = throughput                                                                                          
                                                                                                                                                                                    
        return metrics                                                                                                                                                                
                                                                                                                                                                                    
    def _evaluate_memory(self, model) -> Dict[str, float]:                                                                                                                            
        """                                                                                                                                                                           
        Evaluate model memory usage.                                                                                                                                                  
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model: The model to evaluate                                                                                                                                              
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Dictionary of memory metrics                                                                                                                                              
        """                                                                                                                                                                           
        logger.info("Evaluating memory usage")                                                                                                                                        
                                                                                                                                                                                    
        # Get model size                                                                                                                                                              
        param_size = 0                                                                                                                                                                
        for param in model.parameters():                                                                                                                                              
            param_size += param.nelement() * param.element_size()                                                                                                                     
                                                                                                                                                                                    
        buffer_size = 0                                                                                                                                                               
        for buffer in model.buffers():                                                                                                                                                
            buffer_size += buffer.nelement() * buffer.element_size()                                                                                                                  
                                                                                                                                                                                    
        size_mb = (param_size + buffer_size) / 1024 / 1024                                                                                                                            
                                                                                                                                                                                    
        # Measure GPU memory if available                                                                                                                                             
        gpu_memory_mb = 0                                                                                                                                                             
        if torch.cuda.is_available() and model.device.type == "cuda":                                                                                                                 
            torch.cuda.synchronize()                                                                                                                                                  
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024                                                                                                           
                                                                                                                                                                                    
        return {                                                                                                                                                                      
            "model_size_mb": size_mb,                                                                                                                                                 
            "gpu_memory_mb": gpu_memory_mb,                                                                                                                                           
        }            """
Evaluation module for LLM optimization.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union

import torch
import numpy as np
from tqdm import tqdm

from llm_optimizer.base import OptimizationStage
from llm_optimizer.utils.model import load_model_and_tokenizer
from llm_optimizer.utils.data import load_bundled_dataset, get_bundled_dataset_path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for language models.
    
    Supports various evaluation metrics:
    - Perplexity
    - Completion accuracy
    - Token generation speed
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get("evaluation", {})
        
    def evaluate(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a model on various metrics.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Set model to evaluation mode
        model.eval()
        
        results = {}
        
        # Evaluate perplexity if requested
        if "perplexity" in self.eval_config.get("tasks", ["perplexity"]):
            results["perplexity"] = self.evaluate_perplexity(model, tokenizer)
            
        # Evaluate completion accuracy if requested
        if "completion_accuracy" in self.eval_config.get("tasks", []):
            results["completion_accuracy"] = self.evaluate_completion_accuracy(model, tokenizer)
            
        # Evaluate token generation speed if requested
        if "generation_speed" in self.eval_config.get("tasks", []):
            results["tokens_per_second"] = self.evaluate_generation_speed(model, tokenizer)
            
        return results
        
    def evaluate_perplexity(self, model, tokenizer) -> float:
        """
        Evaluate model perplexity on a text dataset.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            
        Returns:
            Perplexity score (lower is better)
        """
        try:
            # Get dataset path
            dataset_path = self.eval_config.get("dataset_path")
            dataset_name = self.eval_config.get("dataset", "tiny_wikitext")
            
            # Load dataset
            if dataset_path and os.path.exists(dataset_path):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                # Use bundled dataset
                text = load_bundled_dataset(dataset_name)
                
            # Limit text length if specified
            max_length = self.eval_config.get("max_sequence_length", 1024)
            if len(text) > max_length:
                text = text[:max_length]
                
            # Tokenize text
            encodings = tokenizer(text, return_tensors="pt")
            
            # Move to the same device as the model
            input_ids = encodings.input_ids.to(model.device)
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                
                # Get loss
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                else:
                    # If loss is not directly available, calculate it
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Calculate perplexity
                perplexity = torch.exp(loss).item()
                
            logger.info(f"Perplexity: {perplexity:.4f}")
            return perplexity
            
        except Exception as e:
            logger.error(f"Error evaluating perplexity: {e}")
            return float("nan")
            
    def evaluate_completion_accuracy(self, model, tokenizer) -> Dict[str, float]:
        """
        Evaluate model completion accuracy on a prompt-completion dataset.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            
        Returns:
            Dictionary of accuracy metrics
        """
        try:
            # Get dataset path
            dataset_path = self.eval_config.get("completion_dataset_path")
            dataset_name = self.eval_config.get("completion_dataset", "tiny_lambada")
            
            # Load dataset
            if dataset_path and os.path.exists(dataset_path):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    examples = json.load(f)
            else:
                # Use bundled dataset
                examples = load_bundled_dataset(dataset_name)
                
            # Limit number of examples if specified
            max_samples = self.eval_config.get("max_samples")
            if max_samples and len(examples) > max_samples:
                examples = examples[:max_samples]
                
            # Track metrics
            exact_match_count = 0
            token_match_count = 0
            total_examples = len(examples)
            
            # Process each example
            for example in tqdm(examples, desc="Evaluating completions"):
                prompt = example["prompt"]
                expected_completion = example["completion"]
                
                # Generate completion
                completion = self._generate_deterministic_completion(model, tokenizer, prompt, expected_completion)
                
                # Check for exact match
                if completion.strip() == expected_completion.strip():
                    exact_match_count += 1
                    token_match_count += 1
                else:
                    # Check for token overlap
                    expected_tokens = set(tokenizer.tokenize(expected_completion))
                    completion_tokens = set(tokenizer.tokenize(completion))
                    
                    if expected_tokens.intersection(completion_tokens):
                        token_match_count += 1
                
            # Calculate metrics
            exact_match_accuracy = exact_match_count / total_examples if total_examples > 0 else 0
            token_match_accuracy = token_match_count / total_examples if total_examples > 0 else 0
            
            logger.info(f"Completion Accuracy - Exact: {exact_match_accuracy:.4f}, Token: {token_match_accuracy:.4f}")
            
            return {
                "exact_match_accuracy": exact_match_accuracy,
                "token_match_accuracy": token_match_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error evaluating completion accuracy: {e}")
            return {
                "exact_match_accuracy": float("nan"),
                "token_match_accuracy": float("nan")
            }
    
    def _generate_deterministic_completion(self, model, tokenizer, prompt, expected_completion):
        """
        Generate a deterministic completion for a prompt.
        
        This is a simplified approach that doesn't use sampling but instead:
        1. Tokenizes the prompt
        2. Runs a forward pass to get logits
        3. Takes the most likely token at each step
        4. Generates up to the length of the expected completion
        
        Args:
            model: The model to use
            tokenizer: The tokenizer
            prompt: The input prompt
            expected_completion: The expected completion (used for length)
            
        Returns:
            Generated completion text
        """
        try:
            # Tokenize prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            
            # Get expected completion length (in tokens)
            expected_tokens = tokenizer(expected_completion, return_tensors="pt").input_ids.size(1)
            
            # Generate deterministically (no sampling, just greedy)
            with torch.no_grad():
                # For simple models, use greedy generation
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=expected_tokens + 5,  # Add a few extra tokens
                    do_sample=False,  # No sampling
                    num_beams=1,      # No beam search
                    temperature=1.0,  # No temperature
                    top_p=1.0,        # No top-p filtering
                    top_k=0,          # No top-k filtering
                )
                
                # Extract only the generated part (without the prompt)
                generated_ids = output_ids[0, input_ids.size(1):]
                
                # Decode the generated tokens
                completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
            return completion
            
        except Exception as e:
            logger.error(f"Error in deterministic generation: {e}")
            # Return empty string on error
            return ""
            
    def evaluate_generation_speed(self, model, tokenizer) -> float:
        """
        Evaluate token generation speed.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            
        Returns:
            Tokens per second
        """
        try:
            # Use a standard prompt
            prompt = "The quick brown fox jumps over the lazy dog. "
            
            # Tokenize prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            
            # Number of tokens to generate
            num_tokens = self.eval_config.get("generation_tokens", 50)
            
            # Warm-up
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Measure generation time
            start_time = time.time()
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
                
            end_time = time.time()
            
            # Calculate tokens per second
            generated_tokens = output_ids.size(1) - input_ids.size(1)
            generation_time = end_time - start_time
            tokens_per_second = generated_tokens / generation_time
            
            logger.info(f"Generation Speed: {tokens_per_second:.2f} tokens/sec")
            
            return tokens_per_second
            
        except Exception as e:
            logger.error(f"Error evaluating generation speed: {e}")
            return float("nan")


class EvaluationStage(OptimizationStage):
    """
    Evaluation stage for LLM optimization pipeline.
    
    Evaluates models on various metrics and produces reports.
    """
    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the evaluation stage.
        
        Args:
            model_state: Current state of the model
            
        Returns:
            Updated model state with evaluation metrics
        """
        self.validate_input(model_state)
        
        # Extract configuration
        model_path = model_state["model_path"]
        output_dir = os.path.join(
            self.config.get("output_dir", "./outputs"),
            "evaluation"
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.config)
        
        # Run evaluation
        metrics = evaluator.evaluate(model_path)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Return updated model state
        return {
            "model_state": model_state,  # Pass through unchanged
            "metrics": metrics,
            "artifacts": {
                "metrics_path": metrics_path
            }
        }
