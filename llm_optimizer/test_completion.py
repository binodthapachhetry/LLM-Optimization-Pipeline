"""
Test script for manually testing model completions.
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any

import torch
from transformers import set_seed

from llm_optimizer.utils.model import load_model_and_tokenizer
from llm_optimizer.utils.diagnostics import test_model_completion, analyze_tokenizer_behavior
from llm_optimizer.utils.data import load_bundled_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_completion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_single_example(model_path: str, prompt: str, expected_completion: str = None, 
                        max_new_tokens: int = 50, temperature: float = 0.0, 
                        do_sample: bool = False) -> Dict[str, Any]:
    """
    Test a single completion example.
    
    Args:
        model_path: Path to the model
        prompt: The prompt to complete
        expected_completion: Expected completion (optional)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        do_sample: Whether to use sampling
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing completion for model {model_path}")
    logger.info(f"Prompt: {prompt}")
    if expected_completion:
        logger.info(f"Expected completion: {expected_completion}")
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        
        # Analyze tokenizer
        tokenizer_analysis = analyze_tokenizer_behavior(tokenizer, prompt, model_path)
        
        # Set model to evaluation mode
        model.eval()
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Generate with specified parameters
        with torch.no_grad():
            if do_sample:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None
                )
            else:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None
                )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract completion (text after the prompt)
        completion = generated_text[len(prompt):].strip()
        
        # Check if the completion matches the expected completion
        is_match = False
        match_info = {}
        
        if expected_completion:
            # Try different matching criteria
            exact_match = completion == expected_completion
            case_insensitive_match = completion.lower() == expected_completion.lower()
            contains_match = expected_completion.lower() in completion.lower()
            starts_with_match = completion.lower().startswith(expected_completion.lower())
            
            is_match = exact_match or case_insensitive_match
            
            match_info = {
                "exact_match": exact_match,
                "case_insensitive_match": case_insensitive_match,
                "contains_match": contains_match,
                "starts_with_match": starts_with_match
            }
        
        # Log the results
        logger.info(f"Generated text: {generated_text}")
        logger.info(f"Extracted completion: {completion}")
        if expected_completion:
            logger.info(f"Match info: {match_info}")
        
        return {
            "prompt": prompt,
            "expected_completion": expected_completion,
            "generated_text": generated_text,
            "completion": completion,
            "is_match": is_match,
            "match_info": match_info,
            "tokenizer_analysis": tokenizer_analysis
        }
        
    except Exception as e:
        logger.error(f"Error testing completion: {e}")
        return {
            "prompt": prompt,
            "error": str(e)
        }

def test_bundled_dataset(model_path: str, dataset_name: str = "tiny_lambada", 
                         max_examples: int = 5) -> List[Dict[str, Any]]:
    """
    Test completions on a bundled dataset.
    
    Args:
        model_path: Path to the model
        dataset_name: Name of the bundled dataset
        max_examples: Maximum number of examples to test
        
    Returns:
        List of test results
    """
    logger.info(f"Testing completions for model {model_path} on dataset {dataset_name}")
    
    # Load dataset
    try:
        dataset = load_bundled_dataset(dataset_name)
        logger.info(f"Loaded dataset {dataset_name} with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return [{"error": f"Failed to load dataset: {str(e)}"}]
    
    # Limit number of examples
    if max_examples > 0:
        dataset = dataset[:max_examples]
    
    # Test each example
    results = []
    for i, example in enumerate(dataset):
        prompt = example.get("prompt", "")
        expected = example.get("completion", "")
        
        if not prompt:
            logger.warning(f"Skipping example {i} due to missing prompt")
            continue
        
        logger.info(f"Testing example {i+1}/{len(dataset)}")
        
        # Test with different parameters
        for temp in [0.0, 0.7]:
            for do_sample in [False, True]:
                result = test_single_example(
                    model_path=model_path,
                    prompt=prompt,
                    expected_completion=expected,
                    temperature=temp,
                    do_sample=do_sample
                )
                result["example_id"] = i
                result["parameters"] = {
                    "temperature": temp,
                    "do_sample": do_sample
                }
                results.append(result)
    
    # Save results to file
    results_file = f"test_results_{model_path.replace('/', '_')}_{dataset_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved test results to {results_file}")
    
    return results

def main():
    """Main function for the test script."""
    parser = argparse.ArgumentParser(description="Test model completions")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--prompt", type=str, help="Prompt to complete")
    parser.add_argument("--expected", type=str, help="Expected completion")
    parser.add_argument("--dataset", type=str, default="tiny_lambada", help="Bundled dataset to test")
    parser.add_argument("--max-examples", type=int, default=5, help="Maximum number of examples to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    if args.prompt:
        # Test single example
        result = test_single_example(
            model_path=args.model,
            prompt=args.prompt,
            expected_completion=args.expected
        )
        print(json.dumps(result, indent=2, default=str))
    else:
        # Test bundled dataset
        results = test_bundled_dataset(
            model_path=args.model,
            dataset_name=args.dataset,
            max_examples=args.max_examples
        )
        print(f"Tested {len(results)} examples. Results saved to test_results_{args.model.replace('/', '_')}_{args.dataset}.json")

if __name__ == "__main__":
    main()
