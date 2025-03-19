#!/usr/bin/env python
"""
Benchmark script for comparing Llama-3.2-1B-Instruct with its GGUF quantized version.
"""

import os
import argparse
import logging
import json
import time
from typing import Dict, Any

import torch
import matplotlib.pyplot as plt
import numpy as np

from llm_optimizer.utils.model import load_model_and_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Llama-3.2-1B-Instruct against its GGUF version"
    )
    
    parser.add_argument(
        "--hf_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID or path"
    )
    
    parser.add_argument(
        "--gguf_model",
        type=str,
        required=True,
        help="Path to the GGUF model file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="128,512,1024",
        help="Comma-separated list of sequence lengths to test"
    )
    
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,4",
        help="Comma-separated list of batch sizes to test"
    )
    
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations for performance tests"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run benchmarks on"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the difference between machine learning and deep learning in simple terms.",
        help="Prompt to use for generation benchmarks"
    )
    
    return parser.parse_args()

def measure_inference_latency(model, input_ids, num_iterations=10):
    """Measure inference latency for a model."""
    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(input_ids)
            latencies.append((time.time() - start_time) * 1000)  # ms
    
    return {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies)
    }

def measure_generation_latency(model, tokenizer, prompt, num_iterations=5, max_new_tokens=50):
    """Measure text generation latency."""
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, 'device') and hasattr(input_ids, 'to'):
        input_ids = input_ids.to(model.device)
    
    # Warm-up
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids["input_ids"], max_new_tokens=20)
    
    # Measure generation time
    latencies = []
    tokens_per_second = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        output = model.generate(
            input_ids=input_ids["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        end_time = time.time()
        
        # Calculate metrics
        generation_time = end_time - start_time
        new_tokens = output.shape[1] - input_ids["input_ids"].shape[1]
        tokens_per_sec = new_tokens / generation_time
        
        latencies.append(generation_time * 1000)  # ms
        tokens_per_second.append(tokens_per_sec)
    
    return {
        "generation_time_ms": {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies)
        },
        "tokens_per_second": {
            "mean": np.mean(tokens_per_second),
            "median": np.median(tokens_per_second)
        }
    }

def get_model_size(model_path):
    """Get the size of a model file in MB."""
    if os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    elif os.path.isdir(model_path):
        total_size = 0
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    else:
        return 0

def plot_latency_comparison(hf_results, gguf_results, sequence_lengths, output_path):
    """Plot latency comparison between HF and GGUF models."""
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(sequence_lengths))
    width = 0.35
    
    hf_latencies = [hf_results[f"s{seq}"]["mean"] for seq in sequence_lengths]
    gguf_latencies = [gguf_results[f"s{seq}"]["mean"] for seq in sequence_lengths]
    
    plt.bar(x - width/2, hf_latencies, width, label='HuggingFace')
    plt.bar(x + width/2, gguf_latencies, width, label='GGUF')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.title('Inference Latency Comparison')
    plt.xticks(x, sequence_lengths)
    plt.legend()
    
    # Add speedup percentages
    for i, (hf, gguf) in enumerate(zip(hf_latencies, gguf_latencies)):
        speedup = (hf - gguf) / hf * 100
        if speedup > 0:
            plt.text(i + width/2, gguf, f"+{speedup:.1f}%", ha='center', va='bottom', color='green')
        else:
            plt.text(i + width/2, gguf, f"{speedup:.1f}%", ha='center', va='bottom', color='red')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Run the benchmarking process."""
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Running benchmarks on {device}")
    
    # Parse sequence lengths and batch sizes
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model sizes
    hf_size = get_model_size(args.hf_model) if os.path.exists(args.hf_model) else None
    gguf_size = get_model_size(args.gguf_model)
    
    logger.info(f"GGUF model size: {gguf_size:.2f} MB")
    if hf_size:
        logger.info(f"HF model size: {hf_size:.2f} MB")
        logger.info(f"Size reduction: {(hf_size - gguf_size) / hf_size * 100:.2f}%")
    
    # Load models
    logger.info(f"Loading HuggingFace model: {args.hf_model}")
    hf_model, hf_tokenizer = load_model_and_tokenizer(args.hf_model, device_map=device)
    
    logger.info(f"Loading GGUF model: {args.gguf_model}")
    gguf_model, gguf_tokenizer = load_model_and_tokenizer(args.gguf_model, device_map=device)
    
    # Benchmark inference latency
    logger.info("Benchmarking inference latency...")
    hf_latency_results = {}
    gguf_latency_results = {}
    
    for seq_len in sequence_lengths:
        for batch_size in batch_sizes:
            # Skip large configurations that might cause OOM
            if seq_len * batch_size > 8192:
                continue
                
            logger.info(f"Testing sequence length {seq_len}, batch size {batch_size}")
            
            # Create random input
            input_ids = torch.randint(100, 1000, (batch_size, seq_len))
            if hasattr(hf_model, 'device') and hasattr(input_ids, 'to'):
                hf_input = input_ids.to(hf_model.device)
            else:
                hf_input = input_ids
                
            if hasattr(gguf_model, 'device') and hasattr(input_ids, 'to'):
                gguf_input = input_ids.to(gguf_model.device)
            else:
                gguf_input = input_ids
            
            # Measure HF model
            hf_result = measure_inference_latency(hf_model, hf_input, args.num_iterations)
            hf_latency_results[f"b{batch_size}_s{seq_len}"] = hf_result
            
            # Measure GGUF model
            gguf_result = measure_inference_latency(gguf_model, gguf_input, args.num_iterations)
            gguf_latency_results[f"b{batch_size}_s{seq_len}"] = gguf_result
            
            # Log results
            logger.info(f"HF latency: {hf_result['mean']:.2f} ms")
            logger.info(f"GGUF latency: {gguf_result['mean']:.2f} ms")
            speedup = (hf_result['mean'] - gguf_result['mean']) / hf_result['mean'] * 100
            logger.info(f"Speedup: {speedup:.2f}%")
    
    # Benchmark text generation
    logger.info("Benchmarking text generation...")
    try:
        hf_gen_result = measure_generation_latency(hf_model, hf_tokenizer, args.prompt)
        gguf_gen_result = measure_generation_latency(gguf_model, gguf_tokenizer, args.prompt)
        
        logger.info(f"HF generation time: {hf_gen_result['generation_time_ms']['mean']:.2f} ms")
        logger.info(f"GGUF generation time: {gguf_gen_result['generation_time_ms']['mean']:.2f} ms")
        logger.info(f"HF tokens/sec: {hf_gen_result['tokens_per_second']['mean']:.2f}")
        logger.info(f"GGUF tokens/sec: {gguf_gen_result['tokens_per_second']['mean']:.2f}")
    except Exception as e:
        logger.error(f"Error during generation benchmarking: {e}")
        hf_gen_result = None
        gguf_gen_result = None
    
    # Save results
    results = {
        "config": {
            "hf_model": args.hf_model,
            "gguf_model": args.gguf_model,
            "device": device,
            "sequence_lengths": sequence_lengths,
            "batch_sizes": batch_sizes,
            "num_iterations": args.num_iterations,
        },
        "model_size_mb": {
            "huggingface": hf_size,
            "gguf": gguf_size,
            "reduction_percent": (hf_size - gguf_size) / hf_size * 100 if hf_size else None
        },
        "inference_latency": {
            "huggingface": hf_latency_results,
            "gguf": gguf_latency_results
        },
        "generation": {
            "huggingface": hf_gen_result,
            "gguf": gguf_gen_result
        }
    }
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    for seq_len in sequence_lengths:
        hf_seq_results = {f"s{seq_len}": hf_latency_results[f"b1_s{seq_len}"]}
        gguf_seq_results = {f"s{seq_len}": gguf_latency_results[f"b1_s{seq_len}"]}
        plot_path = os.path.join(args.output_dir, f"latency_comparison_s{seq_len}.png")
        plot_latency_comparison(hf_seq_results, gguf_seq_results, [seq_len], plot_path)
    
    # Generate summary report
    summary_path = os.path.join(args.output_dir, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("# Llama-3.2-1B-Instruct vs GGUF Benchmark Summary\n\n")
        
        f.write("## Model Information\n\n")
        f.write(f"- HuggingFace model: {args.hf_model}\n")
        f.write(f"- GGUF model: {args.gguf_model}\n")
        f.write(f"- Device: {device}\n\n")
        
        f.write("## Model Size\n\n")
        if hf_size:
            f.write(f"- HuggingFace: {hf_size:.2f} MB\n")
        f.write(f"- GGUF: {gguf_size:.2f} MB\n")
        if hf_size:
            f.write(f"- Size reduction: {(hf_size - gguf_size) / hf_size * 100:.2f}%\n\n")
        
        f.write("## Inference Latency (Batch Size 1)\n\n")
        f.write("| Sequence Length | HuggingFace (ms) | GGUF (ms) | Speedup |\n")
        f.write("|-----------------|------------------|-----------|--------|\n")
        
        for seq_len in sequence_lengths:
            key = f"b1_s{seq_len}"
            if key in hf_latency_results and key in gguf_latency_results:
                hf_lat = hf_latency_results[key]["mean"]
                gguf_lat = gguf_latency_results[key]["mean"]
                speedup = (hf_lat - gguf_lat) / hf_lat * 100
                f.write(f"| {seq_len} | {hf_lat:.2f} | {gguf_lat:.2f} | {speedup:+.2f}% |\n")
        
        f.write("\n")
        
        if hf_gen_result and gguf_gen_result:
            f.write("## Text Generation\n\n")
            f.write(f"Prompt: \"{args.prompt}\"\n\n")
            
            hf_time = hf_gen_result["generation_time_ms"]["mean"]
            gguf_time = gguf_gen_result["generation_time_ms"]["mean"]
            speedup = (hf_time - gguf_time) / hf_time * 100
            
            f.write(f"- HuggingFace generation time: {hf_time:.2f} ms\n")
            f.write(f"- GGUF generation time: {gguf_time:.2f} ms\n")
            f.write(f"- Speedup: {speedup:+.2f}%\n\n")
            
            hf_tps = hf_gen_result["tokens_per_second"]["mean"]
            gguf_tps = gguf_gen_result["tokens_per_second"]["mean"]
            tps_improvement = (gguf_tps - hf_tps) / hf_tps * 100
            
            f.write(f"- HuggingFace tokens/sec: {hf_tps:.2f}\n")
            f.write(f"- GGUF tokens/sec: {gguf_tps:.2f}\n")
            f.write(f"- Improvement: {tps_improvement:+.2f}%\n")
    
    logger.info(f"Benchmarking complete. Results saved to {args.output_dir}")
    logger.info(f"Summary report: {summary_path}")

if __name__ == "__main__":
    main()
