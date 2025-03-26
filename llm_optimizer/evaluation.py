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
        }            