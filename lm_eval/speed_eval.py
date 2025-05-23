#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import signal
import psutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

# --- Configuration ---
DEFAULT_CONFIG = {
    "models": [
        {
            "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.8},
            "tgi_args": {"max_input_length": 4096, "max_total_tokens": 8192}
        }
    ],
    "benchmark_params": {
        "input_lengths": [128, 512, 1024],
        "output_lengths": [128, 512],
        "batch_sizes": [1, 4, 16],
        "requests_per_batch": 50,
        "concurrent_requests": 8
    },
    "output_dir": "speed_eval_results"
}

# --- Process Management ---
def cleanup_processes(process_name: str) -> None:
    """Terminate any running processes matching process_name"""
    print(f"Cleaning up {process_name} processes...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.cmdline()).lower()
            if process_name.lower() in cmdline:
                print(f"Terminating {process_name} process: {proc.pid}")
                os.kill(proc.pid, signal.SIGTERM)
                # Wait briefly to see if it terminates
                time.sleep(2)
                if psutil.pid_exists(proc.pid):
                    os.kill(proc.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def clear_gpu_memory() -> None:
    """Clear GPU memory between benchmark runs"""
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

# --- Server Management ---
def start_vllm_server(model_name: str, port: int = 8000, **kwargs) -> subprocess.Popen:
    """Start a vLLM server for the specified model"""
    print(f"Starting vLLM server for {model_name} on port {port}...")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--model", model_name,
        "--port", str(port),
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(15)  # Adjust based on typical startup time
    return process

def start_tgi_server(model_name: str, port: int = 8000, **kwargs) -> subprocess.Popen:
    """Start a TGI server for the specified model"""
    print(f"Starting TGI server for {model_name} on port {port}...")
    
    # Base command for TGI
    cmd = [
        "docker", "run", "--gpus", "all", 
        "-p", f"{port}:80", 
        "--rm", 
        "-v", f"{os.path.expanduser('~/.cache/huggingface')}:/data",
        "ghcr.io/huggingface/text-generation-inference:latest",
        "--model-id", model_name
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        cmd.append(f"--{key.replace('_', '-')}={value}")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(30)  # TGI might take longer to initialize
    return process

def stop_server(process: subprocess.Popen) -> None:
    """Gracefully stop a server process"""
    if process is None:
        return
    
    print("Stopping server...")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("Force killing server...")
        process.kill()
    
    # Wait for process to fully terminate
    time.sleep(2)

# --- Benchmark Client ---
def run_benchmark_client(
    backend: str,
    port: int = 8000,
    input_length: int = 128,
    output_length: int = 128,
    batch_size: int = 1,
    requests: int = 50,
    concurrency: int = 1,
) -> Dict:
    """Run benchmark client against the specified server"""
    print(f"Running benchmark: {backend}, input={input_length}, output={output_length}, batch={batch_size}")
    
    if backend == "vllm":
        cmd = [
            "python", "-m", "vllm.entrypoints.benchmark_serving",
            "--host", "localhost",
            "--port", str(port),
            "--tokenizer", "hf-internal-testing/llama-tokenizer",  # Replace with appropriate tokenizer
            "--input-len", str(input_length),
            "--output-len", str(output_length),
            "--batch-size", str(batch_size),
            "--n-requests", str(requests),
            "--concurrency", str(concurrency)
        ]
    elif backend == "tgi":
        cmd = [
            "python", "tgi_benchmark.py",  # You'll need to create this script
            "--host", "localhost",
            "--port", str(port),
            "--input-len", str(input_length),
            "--output-len", str(output_length),
            "--batch-size", str(batch_size),
            "--n-requests", str(requests),
            "--concurrency", str(concurrency)
        ]
    
    # Run benchmark command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to extract metrics
        metrics = parse_benchmark_output(result.stdout, backend)
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {"error": str(e)}

def parse_benchmark_output(output: str, backend: str) -> Dict:
    """Parse benchmark output to extract metrics"""
    metrics = {}
    
    if backend == "vllm":
        # Example parsing for vLLM benchmark output
        # Adjust based on actual output format
        try:
            for line in output.splitlines():
                if "Throughput" in line:
                    metrics["throughput"] = float(line.split(":")[-1].strip().split()[0])
                elif "Time per token" in line:
                    metrics["time_per_token"] = float(line.split(":")[-1].strip().split()[0])
                elif "End-to-end latency" in line:
                    metrics["latency"] = float(line.split(":")[-1].strip().split()[0])
        except Exception as e:
            print(f"Error parsing vLLM output: {e}")
            metrics["parsing_error"] = str(e)
    elif backend == "tgi":
        # Implement TGI output parsing
        # This will depend on your tgi_benchmark.py implementation
        pass
    
    return metrics

# --- Results Management ---
def save_results(
    model: str,
    backend: str,
    results: List[Dict],
    output_dir: str
) -> None:
    """Save benchmark results to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw data as CSV
    model_safe = model.replace("/", "_")
    filename = f"{model_safe}_{backend}_results.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved results to {filepath}")
    
    # Save summary as JSON
    summary = {
        "model": model,
        "backend": backend,
        "avg_throughput": df.get("throughput", pd.Series()).mean(),
        "avg_latency": df.get("latency", pd.Series()).mean(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_file = os.path.join(output_dir, f"{model_safe}_{backend}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

def plot_comparison(
    results_dir: str,
    metric: str = "throughput",
    output_file: Optional[str] = None
) -> None:
    """Generate comparison plots between vLLM and TGI"""
    # Find all result files
    vllm_files = list(Path(results_dir).glob("*_vllm_results.csv"))
    tgi_files = list(Path(results_dir).glob("*_tgi_results.csv"))
    
    models = set([f.stem.split("_vllm")[0] for f in vllm_files])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    for model in models:
        vllm_file = next((f for f in vllm_files if f.stem.startswith(model)), None)
        tgi_file = next((f for f in tgi_files if f.stem.startswith(model)), None)
        
        if vllm_file and tgi_file:
            vllm_df = pd.read_csv(vllm_file)
            tgi_df = pd.read_csv(tgi_file)
            
            # Group by input_length and batch_size
            vllm_grouped = vllm_df.groupby(["input_length", "batch_size"])[metric].mean()
            tgi_grouped = tgi_df.groupby(["input_length", "batch_size"])[metric].mean()
            
            # Plot comparison
            # (Implement specific plotting based on your metrics structure)
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

# --- Main Benchmark Function ---
def run_model_benchmark(
    model_config: Dict,
    benchmark_params: Dict,
    output_dir: str
) -> Dict:
    """Run complete benchmark for a model with both vLLM and TGI backends"""
    model_name = model_config["name"]
    model_results = {"model": model_name, "vllm": [], "tgi": []}
    
    # Make sure no servers are running
    cleanup_processes("vllm")
    cleanup_processes("text-generation-inference")
    
    # Benchmark vLLM
    try:
        vllm_process = start_vllm_server(model_name, **model_config.get("vllm_args", {}))
        
        for input_len in benchmark_params["input_lengths"]:
            for output_len in benchmark_params["output_lengths"]:
                for batch_size in benchmark_params["batch_sizes"]:
                    result = run_benchmark_client(
                        backend="vllm",
                        input_length=input_len,
                        output_length=output_len,
                        batch_size=batch_size,
                        requests=benchmark_params["requests_per_batch"],
                        concurrency=benchmark_params["concurrent_requests"]
                    )
                    
                    result.update({
                        "input_length": input_len,
                        "output_length": output_len,
                        "batch_size": batch_size
                    })
                    
                    model_results["vllm"].append(result)
                    
                    # Brief pause between benchmarks
                    time.sleep(2)
    finally:
        stop_server(vllm_process)
        clear_gpu_memory()
        time.sleep(5)  # Allow time for resources to be released
    
    # Save vLLM results
    save_results(model_name, "vllm", model_results["vllm"], output_dir)
    
    # Benchmark TGI
    try:
        tgi_process = start_tgi_server(model_name, **model_config.get("tgi_args", {}))
        
        for input_len in benchmark_params["input_lengths"]:
            for output_len in benchmark_params["output_lengths"]:
                for batch_size in benchmark_params["batch_sizes"]:
                    result = run_benchmark_client(
                        backend="tgi",
                        input_length=input_len,
                        output_length=output_len,
                        batch_size=batch_size,
                        requests=benchmark_params["requests_per_batch"],
                        concurrency=benchmark_params["concurrent_requests"]
                    )
                    
                    result.update({
                        "input_length": input_len,
                        "output_length": output_len,
                        "batch_size": batch_size
                    })
                    
                    model_results["tgi"].append(result)
                    
                    # Brief pause between benchmarks
                    time.sleep(2)
    finally:
        stop_server(tgi_process)
        clear_gpu_memory()
        time.sleep(5)  # Allow time for resources to be released
    
    # Save TGI results
    save_results(model_name, "tgi", model_results["tgi"], output_dir)
    
    return model_results

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Speed evaluation framework for LLMs")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to benchmark")
    parser.add_argument("--vllm-only", action="store_true", help="Only benchmark vLLM")
    parser.add_argument("--tgi-only", action="store_true", help="Only benchmark TGI")
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, "r") as f:
            config.update(json.load(f))
    
    # Override output directory if specified
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Filter models if specified
    if args.models:
        config["models"] = [m for m in config["models"] if m["name"] in args.models]
    
    # Run benchmarks for each model
    all_results = []
    for model_config in config["models"]:
        model_results = run_model_benchmark(
            model_config,
            config["benchmark_params"],
            config["output_dir"]
        )
        all_results.append(model_results)
    
    # Generate comparison plots
    plot_comparison(
        config["output_dir"],
        metric="throughput",
        output_file=os.path.join(config["output_dir"], "throughput_comparison.png")
    )
    plot_comparison(
        config["output_dir"],
        metric="latency",
        output_file=os.path.join(config["output_dir"], "latency_comparison.png")
    )
    
    print(f"All benchmarks completed. Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
