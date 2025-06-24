import lm_eval
import lm_eval.utils
import os
import json
import gc
import torch
import argparse
import subprocess
import time
import psutil
import signal
import logging
from datetime import datetime
from lm_eval.loggers import EvaluationTracker
from lm_eval.evaluator_copy import simple_evaluate
import random
from lm_eval.tasks import TaskManager, get_task_dict

# Set logging level to reduce verbose output
logging.getLogger("lm_eval").setLevel(logging.WARNING)
logging.getLogger("lm_eval.evaluator").setLevel(logging.WARNING)
logging.getLogger("lm_eval.evaluator_copy").setLevel(logging.WARNING)

# Configuration examples
testing_config = {
    "task_name": "hellaswag",
    "log_samples": True,
    "batch_size": 64,
    "num_fewshot": 0,
}

leaderboard_config = {
    "task_name": "openllm",
    "log_samples": True,
    "batch_size": 16,
    "num_fewshot": 0,
    "limit": None,
}

gsm8k_config = {
    "task_name": "gsm8k",
    "log_samples": True,
    "batch_size": 16,
    "num_fewshot": 5,
    "limit": None,
}

mmlu_config = {
    "task_name": "mmlu",
    "log_samples": True,
    "batch_size": 1,
    "num_fewshot": 0,
    "limit": None,
}


def get_gpu_memory_usage():
    """Get GPU memory usage using nvidia-smi and psutil"""
    gpu_info = {}
    
    # Method 1: nvidia-smi
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_info['nvidia_smi'] = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                gpu_info['nvidia_smi'].append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used_mb': int(parts[2]),
                    'memory_total_mb': int(parts[3]),
                    'memory_free_mb': int(parts[4]),
                    'utilization_percent': int(parts[5])
                })
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError) as e:
        print(f"⚠️ nvidia-smi failed: {e}")
    
    # Method 2: PyTorch CUDA
    if torch.cuda.is_available():
        gpu_info['torch_cuda'] = []
        for i in range(torch.cuda.device_count()):
            gpu_info['torch_cuda'].append({
                'device': i,
                'name': torch.cuda.get_device_name(i),
                'memory_allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                'memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
                'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
            })
    
    return gpu_info


def print_gpu_memory_summary(stage: str):
    """Print a summary of GPU memory usage"""
    print(f"\n🔍 GPU Memory Status - {stage}")
    print("=" * 60)
    
    gpu_info = get_gpu_memory_usage()
    
    if 'nvidia_smi' in gpu_info:
        for gpu in gpu_info['nvidia_smi']:
            used_gb = gpu['memory_used_mb'] / 1024
            total_gb = gpu['memory_total_mb'] / 1024
            free_gb = gpu['memory_free_mb'] / 1024
            util_pct = gpu['utilization_percent']
            
            print(f"GPU {gpu['index']} ({gpu['name']}):")
            print(f"  Memory: {used_gb:.2f}GB / {total_gb:.2f}GB used ({free_gb:.2f}GB free)")
            print(f"  Utilization: {util_pct}%")
    
    if 'torch_cuda' in gpu_info:
        print("\nPyTorch CUDA Memory:")
        for gpu in gpu_info['torch_cuda']:
            alloc_gb = gpu['memory_allocated_mb'] / 1024
            reserved_gb = gpu['memory_reserved_mb'] / 1024
            total_gb = gpu['memory_total_mb'] / 1024
            
            print(f"  GPU {gpu['device']}: {alloc_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")
    
    print("=" * 60)


def clear_memory():
    """Clear memory to prevent OOM issues between evaluations"""
    print("🧹 Clearing memory...")
    
    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Reset memory stats
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"⚠️ Could not reset CUDA stats: {e}")
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
        time.sleep(0.5)
    
    print("✅ Memory clearing completed")


def start_vllm_server(model: str, port: int = 8000, **vllm_kwargs):
    """Start a vLLM server for the given model"""
    print(f"🚀 Starting vLLM server for {model} on port {port}")
    
    # Build command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    
    # Add additional vLLM arguments
    for key, value in vllm_kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Start process
    process = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,  # Create new process group for easier cleanup
    )
    
    return process


def wait_for_server_ready(base_url: str, timeout: int = 300):
    """Wait for vLLM server to be ready"""
    import requests
    
    print(f"⏳ Waiting for server at {base_url} to be ready...")
    
    health_url = base_url.replace('/v1/completions', '/health')
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.RequestException:
            pass
        
        time.sleep(5)
        print(".", end="", flush=True)
    
    print(f"\n❌ Server failed to start within {timeout} seconds")
    return False


def kill_vllm_server(process, model_name: str, port: int = 8000):
    """Kill vLLM server and clean up resources"""
    print(f"🔄 Stopping vLLM server for {model_name}")
    
    # Step 1: Graceful termination
    try:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
            print("✅ Graceful termination successful")
    except subprocess.TimeoutExpired:
        print("⚠️ Graceful termination failed, force killing...")
        
        # Step 2: Force kill
        try:
            if process and process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                print("✅ Force kill successful")
        except (ProcessLookupError, OSError):
            pass
    
    # Step 3: Kill any remaining processes on the port
    try:
        result = subprocess.run(
            f"lsof -ti:{port}", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"✅ Killed remaining process {pid} on port {port}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception as e:
        print(f"⚠️ Error killing by port: {e}")
    
    # Step 4: Clear memory
    clear_memory()
    
    # Step 5: Wait a bit for cleanup
    time.sleep(3)
    
    print("✅ Server cleanup completed")


def get_model_dir(model_name):
    """Convert model name to directory name format"""
    return model_name.replace("/", "__")


def group_evaluate_server_models(
    model_groups: dict,
    tasks: list,
    output_dir: str = "multi_server_eval_output",
    port: int = 8000,
    num_concurrent: int = 16,
    max_retries: int = 3,
    tokenized_requests: bool = False,
    vllm_kwargs: dict = None,
    **eval_kwargs
):
    """
    Evaluate multiple model groups using vLLM servers with group_simple_evaluate optimization.
    
    Args:
        model_groups: Dict of {group_name: [model_names]} 
        tasks: List of task names to evaluate
        output_dir: Output directory for results
        port: Port for vLLM server
        num_concurrent: Number of concurrent requests
        max_retries: Number of retry attempts
        tokenized_requests: Whether to use tokenized requests
        vllm_kwargs: Additional arguments for vLLM server
        **eval_kwargs: Additional arguments for group_simple_evaluate
    """
    
    if vllm_kwargs is None:
        vllm_kwargs = {}
    
    print(f"\n{'='*80}")
    print(f"🚀 Multi-Server Group Evaluation")
    print(f"Groups: {list(model_groups.keys())}")
    print(f"Tasks: {tasks}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    base_url = f"http://localhost:{port}/v1/completions"
    
    # Print initial GPU state
    print_gpu_memory_summary("Initial State")
    
    for group_name, models in model_groups.items():
        print(f"\n🔥 Processing group: {group_name}")
        print(f"Models: {models}")
        
        server_process = None
        group_results = {}
        
        try:
            # Process each model in the group with its own server
            group_results = {}
            
            for model_idx, model_name in enumerate(models):
                print(f"\n🔥 Processing model {model_idx + 1}/{len(models)}: {model_name}")
                
                # Start vLLM server for this specific model
                print_gpu_memory_summary(f"Before starting server for {model_name}")
                server_process = start_vllm_server(model_name, port, **vllm_kwargs)
                
                try:
                    # Wait for server to be ready
                    if not wait_for_server_ready(base_url, timeout=300):
                        raise RuntimeError(f"Server failed to start for {model_name}")
                    
                    print_gpu_memory_summary(f"After starting server for {model_name}")
                    
                    # Prepare model arguments for this specific model
                    model_args = (
                        f"model={model_name},"
                        f"base_url={base_url},"
                        f"num_concurrent={num_concurrent},"
                        f"max_retries={max_retries},"
                        f"tokenized_requests={tokenized_requests},"
                        f"cache_requests=True"
                    )
                    
                    # Run evaluation using simple_evaluate
                    print(f"🎯 Running evaluation for {model_name}...")
                    model_results = lm_eval.simple_evaluate(
                        model="local-completions",
                        model_args=model_args,
                        tasks=tasks,
                        verbosity="WARNING",  # Reduce logging output
                        **eval_kwargs
                    )
                    
                    # Store results with model name
                    if model_results:
                        group_results[model_name] = model_results
                        print(f"📊 Results stored for {model_name}")
                    else:
                        print(f"⚠️ No results returned for {model_name}")
                
                finally:
                    # Always clean up the server for this model
                    if server_process:
                        print_gpu_memory_summary(f"Before killing server for {model_name}")
                        kill_vllm_server(server_process, model_name, port)
                        print_gpu_memory_summary(f"After killing server for {model_name}")
                    
                    # Extra cleanup between models
                    time.sleep(5)
            
            print(f"✅ Group evaluation completed for {group_name}")
            print(f"📊 Group results type: {type(group_results)}")
            if group_results:
                print(f"📊 Group results keys: {list(group_results.keys())}")
                for model_name, results in group_results.items():
                    print(f"📊 Model {model_name} results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            else:
                print("⚠️ Group results is None or empty")
            
            # Store results
            all_results[group_name] = group_results
            
        except Exception as e:
            print(f"❌ Error processing group {group_name}: {e}")
            all_results[group_name] = {"error": str(e)}
            raise e
            
        finally:
            # Always clean up the server
            if server_process:
                print_gpu_memory_summary(f"Before killing server for {group_name}")
                kill_vllm_server(server_process, group_name, port)
                print_gpu_memory_summary(f"After killing server for {group_name}")
            
            # Extra cleanup between groups
            time.sleep(5)
    
    # Save combined results
    combined_results_file = os.path.join(output_dir, "all_results.json")
    with open(combined_results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table
    create_comparison_table(all_results, output_dir)
    
    print(f"\n📊 All results saved to {combined_results_file}")
    print_gpu_memory_summary("Final State")
    
    return all_results


def create_comparison_table(all_results, output_dir):
    """Create a comparison table across all models and groups"""
    print("📊 Creating comparison table...")
    
    comparison_data = []
    
    for group_name, group_results in all_results.items():
        if isinstance(group_results, dict) and "error" not in group_results:
            # group_results is already the dictionary with model names as keys
            for model_name, model_results in group_results.items():
                if isinstance(model_results, dict) and "results" in model_results:
                    for task_name, task_results in model_results["results"].items():
                        # Extract main accuracy metric
                        accuracy = None
                        for metric_name, value in task_results.items():
                            if "acc" in metric_name.lower() and isinstance(value, (int, float)):
                                accuracy = value
                                break
                        
                        comparison_data.append({
                            "group": group_name,
                            "model": model_name,
                            "task": task_name,
                            "accuracy": accuracy,
                            "all_metrics": task_results
                        })
    
    if comparison_data:
        # Create markdown table
        md_content = "# Model Comparison Results\n\n"
        md_content += "| Group | Model | Task | Accuracy |\n"
        md_content += "|-------|-------|------|----------|\n"
        
        for row in comparison_data:
            acc_str = f"{row['accuracy']:.4f}" if row['accuracy'] is not None else "N/A"
            md_content += f"| {row['group']} | {row['model']} | {row['task']} | {acc_str} |\n"
        
        # Save comparison table
        comparison_file = os.path.join(output_dir, "comparison_table.md")
        with open(comparison_file, "w") as f:
            f.write(md_content)
        
        # Save detailed comparison as JSON
        comparison_json = os.path.join(output_dir, "detailed_comparison.json")
        with open(comparison_json, "w") as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"📊 Comparison table saved to {comparison_file}")
        print(f"📊 Detailed comparison saved to {comparison_json}")
    else:
        print("⚠️ No valid results found for comparison table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-server group evaluation using vLLM and group_evaluate optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for testing",
    )
    
    # Server configuration
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM server",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=16,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retry attempts",
    )
    
    # vLLM server arguments
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="multi_server_eval_output",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create model groups - fix syntax error
    model_groups = {
        "deepseek": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B","RedHatAI/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8"]
    }
    
    # vLLM server arguments
    vllm_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    
    print("🔧 Running multi-server evaluation with configuration:")
    print(f"  Model groups: {model_groups}")
    print(f"  Port: {args.port}")
    print(f"  vLLM kwargs: {vllm_kwargs}")

    # Run evaluation
    try:
        results = group_evaluate_server_models(
            model_groups=model_groups,
            tasks=["hellaswag"],  # Pass task names as list
            output_dir=args.output_dir,
            port=args.port,
            num_concurrent=args.num_concurrent,
            max_retries=args.max_retries,
            vllm_kwargs=vllm_kwargs,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            log_samples=True,
        )
        print("✅ Multi-server evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise e
        exit(1) 