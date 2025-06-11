#!/usr/bin/env python3
"""
Simple throughput collector that runs vLLM benchmark_throughput.py as subprocess
and collects results into a structured table.
"""

import subprocess
import re
import json
import os
from typing import List, Dict, Any


def run_throughput_benchmark(
    model: str, input_len: int, output_len: int, max_model_len: int = 12000
) -> Dict[str, Any]:
    """Run a single throughput benchmark and return parsed results."""
    cmd = [
        "python3",
        "vllm/benchmarks/benchmark_throughput.py",
        "--backend",
        "vllm",
        "--model",
        model,
        "--input_len",
        str(input_len),
        "--output_len",
        str(output_len),
        "--num_prompts",
        "1",
        "--max-model-len",
        str(max_model_len),
    ]

    print(f"Running: {model} | {input_len}→{output_len}")

    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Capture output while displaying it
        output_lines = []
        
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                
                # Print line in real-time (preserving progress bars)
                print(f"  {line.rstrip()}")
                
                # Store line for parsing
                output_lines.append(line)
            
            # Wait for process to complete
            return_code = process.wait(timeout=300)
            
            # Combine all output for parsing
            full_output = ''.join(output_lines)
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print(f"  Timeout after 300 seconds")
            return {
                "model": model,
                "input_len": input_len,
                "output_len": output_len,
                "requests_per_sec": None,
                "total_tokens_per_sec": None,
                "input_tokens_per_sec": None,
                "output_tokens_per_sec": None,
                "status": "timeout",
            }

        # Parse throughput metrics from output
        # Expected format: "Throughput: X.XX requests/s, Y.YY total tokens/s, Z.ZZ output tokens/s"
        throughput_match = re.search(
            r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s,\s*([\d.]+)\s*output tokens/s", 
            full_output
        )

        if throughput_match:
            requests_per_sec = float(throughput_match.group(1))
            total_tokens_per_sec = float(throughput_match.group(2))
            output_tokens_per_sec = float(throughput_match.group(3))
            
            # Calculate input tokens/s (total - output)
            input_tokens_per_sec = total_tokens_per_sec - output_tokens_per_sec
            
            return {
                "model": model,
                "input_len": input_len,
                "output_len": output_len,
                "requests_per_sec": requests_per_sec,
                "total_tokens_per_sec": total_tokens_per_sec,
                "input_tokens_per_sec": input_tokens_per_sec,
                "output_tokens_per_sec": output_tokens_per_sec,
                "status": "success",
            }
        else:
            print(f"  Failed to parse throughput from output")
            print(f"  Last few lines of output:")
            for line in output_lines[-5:]:
                print(f"    {line.rstrip()}")
            return {
                "model": model,
                "input_len": input_len,
                "output_len": output_len,
                "requests_per_sec": None,
                "total_tokens_per_sec": None,
                "input_tokens_per_sec": None,
                "output_tokens_per_sec": None,
                "status": "failed",
                "error": "Could not parse throughput",
            }

    except Exception as e:
        print(f"  Error: {e}")
        return {
            "model": model,
            "input_len": input_len,
            "output_len": output_len,
            "requests_per_sec": None,
            "total_tokens_per_sec": None,
            "input_tokens_per_sec": None,
            "output_tokens_per_sec": None,
            "status": "error",
            "error": str(e),
        }


def collect_throughput_results():
    """Collect throughput results for all model/config combinations."""

    # Configuration
    models = [
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    ]

    configs = [(1000, 1000), (10000, 1000)]

    results = []

    # Run all combinations
    for model in models:
        for input_len, output_len in configs:
            result = run_throughput_benchmark(model, input_len, output_len)
            results.append(result)

    return results


def create_throughput_table(results: List[Dict[str, Any]]) -> str:
    """Create a markdown table from throughput results."""

    # Filter successful results
    success_results = [r for r in results if r["status"] == "success"]

    if not success_results:
        return "No successful throughput results to display"

    # Get unique configurations
    configs = sorted(set((r["input_len"], r["output_len"]) for r in success_results))
    models = sorted(set(r["model"] for r in success_results))

    lines = ["# Throughput Benchmark Results\n"]
    
    # Define metrics to display
    metrics = [
        ("requests_per_sec", "Requests per Second"),
        ("total_tokens_per_sec", "Total Tokens per Second"),
        ("input_tokens_per_sec", "Input Tokens per Second"),
        ("output_tokens_per_sec", "Output Tokens per Second"),
    ]
    
    for metric_key, metric_name in metrics:
        lines.append(f"## {metric_name}\n")
        
        # Header
        header = "| Model |"
        separator = "|-------|"
        for input_len, output_len in configs:
            header += f" {input_len}→{output_len} |"
            separator += "--------|"

        lines.append(header)
        lines.append(separator)

        # Data rows
        for model in models:
            row = f"| {model} |"
            for input_len, output_len in configs:
                # Find result for this model/config
                result = next(
                    (
                        r
                        for r in success_results
                        if r["model"] == model
                        and r["input_len"] == input_len
                        and r["output_len"] == output_len
                    ),
                    None,
                )
                if result and result[metric_key] is not None:
                    row += f" {result[metric_key]:.2f} |"
                else:
                    row += " - |"
            lines.append(row)
        
        lines.append("")  # Add blank line between tables

    return "\n".join(lines)


def main():
    """Main function to run benchmarks and create table."""
    print("Collecting throughput benchmark results...")

    results = collect_throughput_results()

    # Save raw results
    with open("throughput_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to throughput_results.json")

    # Create and save table
    table = create_throughput_table(results)
    with open("throughput_comparison.md", "w") as f:
        f.write(table)
    print(f"Comparison table saved to throughput_comparison.md")

    # Print summary
    successful = len([r for r in results if r["status"] == "success"])
    total = len(results)
    print(f"Completed {successful}/{total} benchmarks successfully")
    
    # Print detailed summary of successful results
    success_results = [r for r in results if r["status"] == "success"]
    if success_results:
        print(f"\nSummary of successful benchmarks:")
        for result in success_results:
            print(f"  {result['model']} ({result['input_len']}→{result['output_len']}):")
            print(f"    Requests/s: {result['requests_per_sec']:.2f}")
            print(f"    Total tokens/s: {result['total_tokens_per_sec']:.2f}")
            print(f"    Input tokens/s: {result['input_tokens_per_sec']:.2f}")
            print(f"    Output tokens/s: {result['output_tokens_per_sec']:.2f}")

    # Print the table
    print("\n" + table)


if __name__ == "__main__":
    main()

