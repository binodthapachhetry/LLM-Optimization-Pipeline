import os                                                                                                                                                                             
import logging                                                                                                                                                                        
from typing import Dict, Any, Optional, List                                                                                                                                          
                                                                                                                                                                                    
import dspy                                                                                                                                                                           
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch                                                                                                        
from dspy.evaluate import Evaluate                                                                                                                                                    
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage                                                                                                                                      
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                   
class PromptOptimizationStage(OptimizationStage):                                                                                                                                     
    """                                                                                                                                                                               
    Prompt optimization stage for LLM optimization.                                                                                                                                   
                                                                                                                                                                                    
    Uses DSPy to optimize prompts for better performance on specific tasks.                                                                                                           
    Supports various prompt optimization techniques:                                                                                                                                  
    - Few-shot learning                                                                                                                                                               
    - Bootstrapping                                                                                                                                                                   
    - Chain-of-thought prompting                                                                                                                                                      
    - Self-consistency                                                                                                                                                                
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                     
        """                                                                                                                                                                           
        Run the prompt optimization stage.                                                                                                                                            
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_state: Current state of the model                                                                                                                                   
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Updated model state and metrics                                                                                                                                           
        """                                                                                                                                                                           
        self.validate_input(model_state)                                                                                                                                              
                                                                                                                                                                                    
        # Extract configuration                                                                                                                                                       
        model_path = model_state["model_path"]                                                                                                                                        
        method = self.config.get("method", "bootstrap_few_shot")                                                                                                                      
        task = self.config.get("task", "qa")                                                                                                                                          
        output_dir = os.path.join(                                                                                                                                                    
            self.config.get("output_dir", "./outputs"),                                                                                                                               
            f"prompt_optimized_{task}"                                                                                                                                                
        )                                                                                                                                                                             
                                                                                                                                                                                    
        logger.info(f"Optimizing prompts for model {model_path} on task {task}")                                                                                                      
                                                                                                                                                                                    
        # Initialize DSPy with the model                                                                                                                                              
        self._initialize_dspy(model_path)                                                                                                                                             
                                                                                                                                                                                    
        # Load task-specific dataset                                                                                                                                                  
        train_data, eval_data = self._load_task_dataset(task)                                                                                                                         
                                                                                                                                                                                    
        # Define the task module based on the task type                                                                                                                               
        task_module = self._create_task_module(task)                                                                                                                                  
                                                                                                                                                                                    
        # Apply prompt optimization method                                                                                                                                            
        if method == "bootstrap_few_shot":                                                                                                                                            
            optimized_module = self._bootstrap_few_shot(task_module, train_data, eval_data)                                                                                           
        elif method == "bootstrap_random_search":                                                                                                                                     
            optimized_module = self._bootstrap_random_search(task_module, train_data, eval_data)                                                                                      
        elif method == "chain_of_thought":                                                                                                                                            
            optimized_module = self._chain_of_thought(task_module, train_data, eval_data)                                                                                             
        else:                                                                                                                                                                         
            raise ValueError(f"Unsupported prompt optimization method: {method}")                                                                                                     
                                                                                                                                                                                    
        # Evaluate the optimized module                                                                                                                                               
        metrics = self._evaluate_optimized_module(optimized_module, eval_data)                                                                                                        
                                                                                                                                                                                    
        # Save the optimized prompts                                                                                                                                                  
        os.makedirs(output_dir, exist_ok=True)                                                                                                                                        
        optimized_prompts = self._extract_optimized_prompts(optimized_module)                                                                                                         
        self._save_optimized_prompts(optimized_prompts, output_dir)                                                                                                                   
                                                                                                                                                                                    
        logger.info(f"Prompt optimization complete. Results saved to {output_dir}")                                                                                                   
                                                                                                                                                                                    
        # Return updated model state and metrics                                                                                                                                      
        return {                                                                                                                                                                      
            "model_state": {                                                                                                                                                          
                "model_path": model_path,                                                                                                                                             
                "optimized_prompts_path": output_dir,                                                                                                                                 
                "task": task,                                                                                                                                                         
            },                                                                                                                                                                        
            "metrics": metrics,                                                                                                                                                       
            "artifacts": {                                                                                                                                                            
                "optimized_prompts_path": output_dir,                                                                                                                                 
            }                                                                                                                                                                         
        }                                                                                                                                                                             
                                                                                                                                                                                    
    def _initialize_dspy(self, model_path: str) -> None:                                                                                                                              
        """Initialize DSPy with the specified model."""                                                                                                                               
        # Configure DSPy to use the model                                                                                                                                             
        if "openai" in model_path.lower():                                                                                                                                            
            # For OpenAI models                                                                                                                                                       
            api_key = self.config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))                                                                                             
            if not api_key:                                                                                                                                                           
                raise ValueError("OpenAI API key is required for OpenAI models")                                                                                                      
                                                                                                                                                                                    
            dspy.settings.configure(lm=dspy.OpenAI(model=model_path, api_key=api_key))                                                                                                
        else:                                                                                                                                                                         
            # For local models                                                                                                                                                        
            dspy.settings.configure(lm=dspy.HFLocalLM(model_path))                                                                                                                    
                                                                                                                                                                                    
    def _load_task_dataset(self, task: str) -> tuple:                                                                                                                                 
        """Load dataset for the specified task."""                                                                                                                                    
        # Get dataset configuration                                                                                                                                                   
        dataset_name = self.config.get("dataset_name")                                                                                                                                
        dataset_config = self.config.get("dataset_config")                                                                                                                            
        train_split = self.config.get("train_split", "train")                                                                                                                         
        eval_split = self.config.get("eval_split", "validation")                                                                                                                      
                                                                                                                                                                                    
        if task == "qa":                                                                                                                                                              
            # Load QA dataset                                                                                                                                                         
            if not dataset_name:                                                                                                                                                      
                dataset_name = "squad"                                                                                                                                                
                                                                                                                                                                                    
            train_data = dspy.datasets.get(dataset_name, split=train_split, config=dataset_config)                                                                                    
            eval_data = dspy.datasets.get(dataset_name, split=eval_split, config=dataset_config)                                                                                      
                                                                                                                                                                                    
        elif task == "summarization":                                                                                                                                                 
            # Load summarization dataset                                                                                                                                              
            if not dataset_name:                                                                                                                                                      
                dataset_name = "cnn_dailymail"                                                                                                                                        
                dataset_config = "3.0.0"                                                                                                                                              
                                                                                                                                                                                    
            train_data = dspy.datasets.get(dataset_name, split=train_split, config=dataset_config)                                                                                    
            eval_data = dspy.datasets.get(dataset_name, split=eval_split, config=dataset_config)                                                                                      
                                                                                                                                                                                    
        elif task == "classification":                                                                                                                                                
            # Load classification dataset                                                                                                                                             
            if not dataset_name:                                                                                                                                                      
                dataset_name = "glue"                                                                                                                                                 
                dataset_config = "sst2"                                                                                                                                               
                                                                                                                                                                                    
            train_data = dspy.datasets.get(dataset_name, split=train_split, config=dataset_config)                                                                                    
            eval_data = dspy.datasets.get(dataset_name, split=eval_split, config=dataset_config)                                                                                      
                                                                                                                                                                                    
        else:                                                                                                                                                                         
            raise ValueError(f"Unsupported task: {task}")                                                                                                                             
                                                                                                                                                                                    
        # Limit dataset size for efficiency if specified                                                                                                                              
        max_train_examples = self.config.get("max_train_examples")                                                                                                                    
        max_eval_examples = self.config.get("max_eval_examples")                                                                                                                      
                                                                                                                                                                                    
        if max_train_examples and len(train_data) > max_train_examples:                                                                                                               
            train_data = train_data[:max_train_examples]                                                                                                                              
                                                                                                                                                                                    
        if max_eval_examples and len(eval_data) > max_eval_examples:                                                                                                                  
            eval_data = eval_data[:max_eval_examples]                                                                                                                                 
                                                                                                                                                                                    
        return train_data, eval_data                                                                                                                                                  
                                                                                                                                                                                    
    def _create_task_module(self, task: str) -> dspy.Module:                                                                                                                          
        """Create a DSPy module for the specified task."""                                                                                                                            
        if task == "qa":                                                                                                                                                              
            # Define a QA module                                                                                                                                                      
            class QuestionAnswerer(dspy.Module):                                                                                                                                      
                def __init__(self):                                                                                                                                                   
                    super().__init__()                                                                                                                                                
                    self.generate_answer = dspy.ChainOfThought("context, question -> answer")                                                                                         
                                                                                                                                                                                    
                def forward(self, context, question):                                                                                                                                 
                    return self.generate_answer(context=context, question=question)                                                                                                   
                                                                                                                                                                                    
            return QuestionAnswerer()                                                                                                                                                 
                                                                                                                                                                                    
        elif task == "summarization":                                                                                                                                                 
            # Define a summarization module                                                                                                                                           
            class Summarizer(dspy.Module):                                                                                                                                            
                def __init__(self):                                                                                                                                                   
                    super().__init__()                                                                                                                                                
                    self.generate_summary = dspy.Predict("document -> summary")                                                                                                       
                                                                                                                                                                                    
                def forward(self, document):                                                                                                                                          
                    return self.generate_summary(document=document)                                                                                                                   
                                                                                                                                                                                    
            return Summarizer()                                                                                                                                                       
                                                                                                                                                                                    
        elif task == "classification":                                                                                                                                                
            # Define a classification module                                                                                                                                          
            class Classifier(dspy.Module):                                                                                                                                            
                def __init__(self):                                                                                                                                                   
                    super().__init__()                                                                                                                                                
                    self.classify = dspy.Predict("text -> label")                                                                                                                     
                                                                                                                                                                                    
                def forward(self, text):                                                                                                                                              
                    return self.classify(text=text)                                                                                                                                   
                                                                                                                                                                                    
            return Classifier()                                                                                                                                                       
                                                                                                                                                                                    
        else:                                                                                                                                                                         
            raise ValueError(f"Unsupported task: {task}")                                                                                                                             
                                                                                                                                                                                    
    def _bootstrap_few_shot(self, module, train_data, eval_data):                                                                                                                     
        """Apply bootstrap few-shot optimization."""                                                                                                                                  
        # Define metric                                                                                                                                                               
        metric = self._get_task_metric()                                                                                                                                              
                                                                                                                                                                                    
        # Create optimizer                                                                                                                                                            
        num_bootstrapping_examples = self.config.get("num_bootstrapping_examples", 3)                                                                                                 
        optimizer = BootstrapFewShot(metric=metric, num_examples=num_bootstrapping_examples)                                                                                          
                                                                                                                                                                                    
        # Optimize the module                                                                                                                                                         
        optimized_module = optimizer.optimize(module, trainset=train_data, valset=eval_data)                                                                                          
                                                                                                                                                                                    
        return optimized_module                                                                                                                                                       
                                                                                                                                                                                    
    def _bootstrap_random_search(self, module, train_data, eval_data):                                                                                                                
        """Apply bootstrap few-shot with random search optimization."""                                                                                                               
        # Define metric                                                                                                                                                               
        metric = self._get_task_metric()                                                                                                                                              
                                                                                                                                                                                    
        # Create optimizer                                                                                                                                                            
        num_bootstrapping_examples = self.config.get("num_bootstrapping_examples", 3)                                                                                                 
        num_candidates = self.config.get("num_candidates", 5)                                                                                                                         
                                                                                                                                                                                    
        optimizer = BootstrapFewShotWithRandomSearch(                                                                                                                                 
            metric=metric,                                                                                                                                                            
            num_examples=num_bootstrapping_examples,                                                                                                                                  
            num_candidates=num_candidates                                                                                                                                             
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Optimize the module                                                                                                                                                         
        optimized_module = optimizer.optimize(module, trainset=train_data, valset=eval_data)                                                                                          
                                                                                                                                                                                    
        return optimized_module                                                                                                                                                       
                                                                                                                                                                                    
    def _chain_of_thought(self, module, train_data, eval_data):                                                                                                                       
        """Apply chain-of-thought prompting optimization."""                                                                                                                          
        # For chain-of-thought, we need to ensure the module uses ChainOfThought                                                                                                      
        # This is typically handled in the module definition                                                                                                                          
                                                                                                                                                                                    
        # Define metric                                                                                                                                                               
        metric = self._get_task_metric()                                                                                                                                              
                                                                                                                                                                                    
        # Create optimizer (similar to bootstrap)                                                                                                                                     
        num_examples = self.config.get("num_examples", 3)                                                                                                                             
        optimizer = BootstrapFewShot(metric=metric, num_examples=num_examples)                                                                                                        
                                                                                                                                                                                    
        # Optimize the module                                                                                                                                                         
        optimized_module = optimizer.optimize(module, trainset=train_data, valset=eval_data)                                                                                          
                                                                                                                                                                                    
        return optimized_module                                                                                                                                                       
                                                                                                                                                                                    
    def _get_task_metric(self):                                                                                                                                                       
        """Get the appropriate metric for the current task."""                                                                                                                        
        task = self.config.get("task", "qa")                                                                                                                                          
                                                                                                                                                                                    
        if task == "qa":                                                                                                                                                              
            return dspy.evaluate.answer_exact_match                                                                                                                                   
        elif task == "summarization":                                                                                                                                                 
            return dspy.evaluate.rouge                                                                                                                                                
        elif task == "classification":                                                                                                                                                
            return dspy.evaluate.accuracy                                                                                                                                             
        else:                                                                                                                                                                         
            raise ValueError(f"Unsupported task: {task}")                                                                                                                             
                                                                                                                                                                                    
    def _evaluate_optimized_module(self, module, eval_data):                                                                                                                          
        """Evaluate the optimized module on the evaluation dataset."""                                                                                                                
        # Create evaluator                                                                                                                                                            
        metric = self._get_task_metric()                                                                                                                                              
        evaluator = Evaluate(metric=metric)                                                                                                                                           
                                                                                                                                                                                    
        # Run evaluation                                                                                                                                                              
        evaluation_result = evaluator.evaluate(module, eval_data)                                                                                                                     
                                                                                                                                                                                    
        # Extract metrics                                                                                                                                                             
        metrics = {                                                                                                                                                                   
            "score": evaluation_result.score,                                                                                                                                         
            "num_examples": len(eval_data),                                                                                                                                           
        }                                                                                                                                                                             
                                                                                                                                                                                    
        # Add detailed metrics if available                                                                                                                                           
        if hasattr(evaluation_result, "detailed_metrics"):                                                                                                                            
            metrics.update(evaluation_result.detailed_metrics)                                                                                                                        
                                                                                                                                                                                    
        return metrics                                                                                                                                                                
                                                                                                                                                                                    
    def _extract_optimized_prompts(self, module):                                                                                                                                     
        """Extract the optimized prompts from the module."""                                                                                                                          
        # This will depend on the specific DSPy module structure                                                                                                                      
        optimized_prompts = {}                                                                                                                                                        
                                                                                                                                                                                    
        # Extract prompts from each predictor in the module                                                                                                                           
        for name, predictor in module.__dict__.items():                                                                                                                               
            if isinstance(predictor, dspy.Predict) or isinstance(predictor, dspy.ChainOfThought):                                                                                     
                optimized_prompts[name] = predictor.prompt.template                                                                                                                   
                                                                                                                                                                                    
        return optimized_prompts                                                                                                                                                      
                                                                                                                                                                                    
    def _save_optimized_prompts(self, prompts, output_dir):                                                                                                                           
        """Save the optimized prompts to files."""                                                                                                                                    
        import json                                                                                                                                                                   
                                                                                                                                                                                    
        # Save all prompts to a single JSON file                                                                                                                                      
        prompts_path = os.path.join(output_dir, "optimized_prompts.json")                                                                                                             
        with open(prompts_path, "w") as f:                                                                                                                                            
            json.dump(prompts, f, indent=2)                                                                                                                                           
                                                                                                                                                                                    
        # Also save each prompt to a separate file for easy access                                                                                                                    
        for name, prompt in prompts.items():                                                                                                                                          
            prompt_path = os.path.join(output_dir, f"{name}.txt")                                                                                                                     
            with open(prompt_path, "w") as f:                                                                                                                                         
                f.write(prompt) 