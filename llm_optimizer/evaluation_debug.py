"""
Debug version of the ModelEvaluator with enhanced logging for troubleshooting.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_optimizer.utils.diagnostics import log_completion_evaluation, test_model_completion
from llm_optimizer.utils.model import load_model_and_tokenizer

logger = logging.getLogger(__name__)

class DebugModelEvaluator:
    """
    Debug version of ModelEvaluator with enhanced logging for troubleshooting.
    """
    
    def __init__(self, config):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_config = config.get("evaluation", {})
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("evaluation_debug.log"),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Initialized DebugModelEvaluator with config:")
        logger.info(json.dumps(self.evaluation_config, indent=2, default=str))
    
    def evaluate(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a model on various metrics with enhanced debugging.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Starting evaluation of model: {model_path}")
        
        # Load model and tokenizer
        try:
            model, tokenizer = load_model_and_tokenizer(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            
            # Analyze tokenizer behavior
            from llm_optimizer.utils.diagnostics import analyze_tokenizer_behavior
            tokenizer_analysis = analyze_tokenizer_behavior(
                tokenizer, 
                "This is a test sentence to analyze tokenizer behavior.",
                tokenizer_name=model_path
            )
            logger.info(f"Tokenizer analysis: {json.dumps(tokenizer_analysis, indent=2, default=str)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"error": f"Model loading failed: {str(e)}"}
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize results dictionary
        results = {}
        
        # Evaluate completion accuracy with detailed logging
        try:
            completion_results = self._evaluate_completion_accuracy_debug(model, tokenizer, model_path)
            results.update(completion_results)
        except Exception as e:
            logger.error(f"Completion accuracy evaluation failed: {e}")
            results["completion_accuracy_error"] = str(e)
        
        # Add other evaluation metrics as needed
        
        logger.info(f"Evaluation completed for {model_path}")
        logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")
        
        return results
    
    def _evaluate_completion_accuracy_debug(self, model, tokenizer, model_name: str) -> Dict[str, Any]:
        """
        Evaluate completion accuracy with detailed debugging.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            model_name: Name or identifier of the model
            
        Returns:
            Dictionary of completion accuracy metrics
        """
        logger.info(f"Evaluating completion accuracy for {model_name}")
        
        # Load completion dataset
        completion_dataset_path = self.evaluation_config.get("completion_dataset_path")
        
        if not completion_dataset_path:
            logger.warning("No completion dataset path specified")
            return {"completion_accuracy": "No dataset specified"}
        
        try:
            with open(completion_dataset_path, 'r') as f:
                completion_data = json.load(f)
            logger.info(f"Loaded completion dataset from {completion_dataset_path} with {len(completion_data)} examples")
            
            # Log the dataset content
            logger.info(f"Dataset content: {json.dumps(completion_data[:5], indent=2)}")
        except Exception as e:
            logger.error(f"Failed to load completion dataset: {e}")
            return {"completion_accuracy_error": f"Dataset loading failed: {str(e)}"}
        
        # Initialize counters
        exact_matches = 0
        token_matches = 0
        case_insensitive_matches = 0
        contains_matches = 0
        starts_with_matches = 0
        total_examples = 0
        
        # Process each example with detailed logging
        all_results = []
        
        for i, example in enumerate(completion_data):
            if i >= self.evaluation_config.get("max_samples", 10):
                logger.info(f"Reached max_samples limit of {self.evaluation_config.get('max_samples', 10)}")
                break
                
            prompt = example.get("prompt", "")
            expected = example.get("completion", "")
            
            if not prompt or not expected:
                logger.warning(f"Skipping example {i} due to missing prompt or completion")
                continue
            
            logger.info(f"\n{'='*80}\nEvaluating example {i+1}/{len(completion_data)}\n{'='*80}")
            logger.info(f"Prompt: \"{prompt}\"")
            logger.info(f"Expected completion: \"{expected}\"")
            
            # Test with different generation parameters
            for temp in [0.0, 0.3, 0.7]:
                for do_sample in [False, True]:
                    for max_tokens in [20, 50]:
                        try:
                            # Prepare generation parameters
                            gen_params = {
                                "temperature": temp,
                                "do_sample": do_sample,
                                "max_new_tokens": max_tokens,
                                "model_name": f"{model_name} (temp={temp}, do_sample={do_sample}, max_tokens={max_tokens})"
                            }
                            
                            logger.info(f"Testing with parameters: {gen_params}")
                            
                            # Tokenize input
                            inputs = tokenizer(prompt, return_tensors="pt")
                            input_ids = inputs["input_ids"].to(model.device)
                            
                            # Generate text
                            with torch.no_grad():
                                if do_sample:
                                    outputs = model.generate(
                                        input_ids,
                                        max_new_tokens=max_tokens,
                                        do_sample=True,
                                        temperature=temp,
                                        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None
                                    )
                                else:
                                    outputs = model.generate(
                                        input_ids,
                                        max_new_tokens=max_tokens,
                                        do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None
                                    )
                            
                            # Decode generated text
                            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            
                            # Try different ways to extract the completion
                            # Method 1: Everything after the prompt
                            completion1 = generated_text[len(prompt):].strip()
                            
                            # Method 2: Try to find the expected completion anywhere in the generated text
                            completion2 = ""
                            if expected.lower() in generated_text.lower():
                                idx = generated_text.lower().find(expected.lower())
                                completion2 = generated_text[idx:idx+len(expected)]
                            
                            # Check different matching criteria
                            exact_match = completion1 == expected
                            case_insensitive_match = completion1.lower() == expected.lower()
                            contains_match = expected.lower() in generated_text.lower()
                            starts_with_match = completion1.lower().startswith(expected.lower())
                            
                            # Log detailed results
                            match_info = {
                                "exact_match": exact_match,
                                "case_insensitive_match": case_insensitive_match,
                                "contains_match": contains_match,
                                "starts_with_match": starts_with_match,
                                "generation_params": gen_params
                            }
                            
                            log_completion_evaluation(
                                prompt=prompt,
                                expected_completion=expected,
                                generated_text=generated_text,
                                extracted_completion=completion1,
                                is_match=exact_match or case_insensitive_match,
                                model_name=gen_params["model_name"],
                                additional_info=match_info
                            )
                            
                            # Store results for this configuration
                            all_results.append({
                                "example_id": i,
                                "prompt": prompt,
                                "expected": expected,
                                "generated": generated_text,
                                "completion1": completion1,
                                "completion2": completion2,
                                "exact_match": exact_match,
                                "case_insensitive_match": case_insensitive_match,
                                "contains_match": contains_match,
                                "starts_with_match": starts_with_match,
                                "generation_params": gen_params
                            })
                            
                            # Update counters (using the best result for this example)
                            if exact_match:
                                exact_matches += 1
                            if case_insensitive_match:
                                case_insensitive_matches += 1
                            if contains_match:
                                contains_matches += 1
                            if starts_with_match:
                                starts_with_matches += 1
                                
                        except Exception as e:
                            logger.error(f"Error generating completion for example {i}: {e}")
                            all_results.append({
                                "example_id": i,
                                "error": str(e),
                                "generation_params": gen_params
                            })
            
            total_examples += 1
        
        # Calculate metrics
        if total_examples > 0:
            exact_match_accuracy = exact_matches / total_examples
            case_insensitive_accuracy = case_insensitive_matches / total_examples
            contains_accuracy = contains_matches / total_examples
            starts_with_accuracy = starts_with_matches / total_examples
        else:
            exact_match_accuracy = 0.0
            case_insensitive_accuracy = 0.0
            contains_accuracy = 0.0
            starts_with_accuracy = 0.0
        
        # Log summary
        logger.info(f"\n{'='*80}\nCOMPLETION ACCURACY SUMMARY FOR {model_name}\n{'='*80}")
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Exact matches: {exact_matches} ({exact_match_accuracy:.2%})")
        logger.info(f"Case-insensitive matches: {case_insensitive_matches} ({case_insensitive_accuracy:.2%})")
        logger.info(f"Contains matches: {contains_matches} ({contains_accuracy:.2%})")
        logger.info(f"Starts-with matches: {starts_with_matches} ({starts_with_accuracy:.2%})")
        
        # Save detailed results to file
        results_file = f"completion_results_{model_name.replace('/', '_')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved detailed results to {results_file}")
        
        return {
            "completion_accuracy": {
                "exact_match_accuracy": exact_match_accuracy,
                "case_insensitive_accuracy": case_insensitive_accuracy,
                "contains_accuracy": contains_accuracy,
                "starts_with_accuracy": starts_with_accuracy,
                "total_examples": total_examples,
                "exact_matches": exact_matches,
                "case_insensitive_matches": case_insensitive_matches,
                "contains_matches": contains_matches,
                "starts_with_matches": starts_with_matches,
                "detailed_results_file": results_file
            }
        }

def evaluate_model_debug(model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model with debug logging.
    
    Args:
        model_path: Path to the model
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = DebugModelEvaluator(config)
    return evaluator.evaluate(model_path)
