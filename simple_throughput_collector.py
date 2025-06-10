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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse throughput from output
        throughput_match = re.search(
            r"Throughput:\s*([\d.]+)\s*requests/s", result.stdout
        )

        if throughput_match:
            throughput = float(throughput_match.group(1))
            return {
                "model": model,
                "input_len": input_len,
                "output_len": output_len,
                "throughput": throughput,
                "status": "success",
            }
        else:
            print(f"  Failed to parse throughput from output")
            return {
                "model": model,
                "input_len": input_len,
                "output_len": output_len,
                "throughput": None,
                "status": "failed",
                "error": "Could not parse throughput",
            }

    except subprocess.TimeoutExpired:
        print(f"  Timeout after 300 seconds")
        return {
            "model": model,
            "input_len": input_len,
            "output_len": output_len,
            "throughput": None,
            "status": "timeout",
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {
            "model": model,
            "input_len": input_len,
            "output_len": output_len,
            "throughput": None,
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

    # Create table
    lines = ["# Throughput Benchmark Results\n"]
    lines.append("**Units:** Requests per second\n")

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
            if result and result["throughput"]:
                row += f" {result['throughput']:.2f} |"
            else:
                row += " - |"
        lines.append(row)

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

    # Print the table
    print("\n" + table)


if __name__ == "__main__":
    main()

