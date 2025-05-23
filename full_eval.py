import lm_eval
import lm_eval.utils
from flip_eval import FlipEval
from KLEval import KLEval
import torch
import gc
import os
import re
import json
import shutil
import signal
import psutil
import multiprocessing
from datetime import datetime
from lm_eval.loggers import EvaluationTracker

### Run simple test on hellaswag
model_groups = [
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
     "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8",
     "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16"),
    ("meta-llama/Meta-Llama-3-8B-Instruct",
     "RedHatAI/Meta-Llama-3-8B-Instruct-quantized.w8a8",
     "RedHatAI/Meta-Llama-3-8B-Instruct-quantized.w4a16"),
]

# Base output directory for all model outputs
INPUT_DIR = "/home/spooky/Documents/smol_projects/inference/lm-evaluation-harness/auto_output"
OUTPUT_DIR = "auto_output"  # Top-level directory for all organized outputs

### Configs for MMLU, ARC-C, GSM8kCOT, HellaSwag, Winogrande, TruthfulQA

hellaswag_config = {
    "task_name": "hellaswag",
    "log_samples": True,
    "batch_size": 16,
}

mmlu_config = {
    "task_name": "mmlu",
    "log_samples": True,
    "batch_size": 16,
}

arc_c_config = {
    "task_name": "arc_challenge",
    "log_samples": True,
    "batch_size": 16,
}

gsm8kcot_config = {
    "task_name": "gsm8k",
    "log_samples": True,
    "batch_size": 16,
}

winogrande_config = {
    "task_name": "winogrande",
    "log_samples": True,
    "batch_size": 16,
}

truthfulqa_config = {
    "task_name": "truthfulqa",
    "log_samples": True,
    "batch_size": 16,
}

def clear_memory():
    # Delete any existing models from memory
    try:
        del model
    except:
        pass
    
    # Clear all unused PyTorch memory caches
    torch.cuda.empty_cache()
    
    # Clear CPU tensor caches if any exist in global namespace
    for name in list(globals()):
        if isinstance(globals()[name], torch.Tensor):
            del globals()[name]
    
    # Force garbage collection multiple times to ensure tensors are freed
    gc.collect()
    gc.collect()
    
    # Additional memory info (optional)
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
    
    # Report on CPU memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        print(f"CPU Memory: {process.memory_info().rss/1e9:.2f} GB")
    except ImportError:
        print("Install psutil to monitor CPU memory usage")

def cleanup_vllm_processes():
    """Kill any existing vLLM processes to prevent conflicts"""
    current_process = psutil.Process()
    
    # Get all child processes
    for proc in psutil.process_iter():
        try:
            # Check if this is a Python process
            if "python" in proc.name().lower():
                # Check if "vllm" appears in the command line
                cmdline = " ".join(proc.cmdline()).lower()
                if "vllm" in cmdline and proc.pid != current_process.pid and proc.pid not in [p.pid for p in current_process.children(recursive=True)]:
                    print(f"Terminating previous vLLM process: {proc.pid}")
                    try:
                        proc.terminate()
                        # Wait for up to 3 seconds
                        gone, still_alive = psutil.wait_procs([proc], timeout=3)
                        for p in still_alive:
                            # If still alive after 3 seconds, kill it
                            p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def get_model_dir(model_name):
    """Convert model name to directory name format used in output folder"""
    # Replace slashes with double underscores
    return model_name.replace("/", "__")

def extract_base_model_name(model_name):
    """Extract base model name without version details for group naming"""
    # Look for parameter count like 7B, 1.5B, etc.
    param_match = re.search(r'(\d+\.?\d*)B', model_name)
    param_count = param_match.group(0) if param_match else ""
    
    # Use regex to extract main model family name
    match = re.search(r"([^/]+)/([^-]+)", model_name)
    if match:
        vendor, model = match.groups()
        return f"{vendor}_{model}_{param_count}".lower().rstrip('_')
    
    # Fallback to just the last part of the model name
    parts = model_name.split("/")
    if len(parts) > 1:
        return f"{parts[-1]}_{param_count}".lower().rstrip('_')
    
    return model_name.lower()

def create_directory_structure(model_groups):
    """Create the directory structure for output organization"""
    
    # Create top-level output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create group directories and model directories
    group_dirs = {}
    for models in model_groups:
        base_model = models[0]
        base_model_name = extract_base_model_name(base_model)
        group_dir_name = f"model_group_{base_model_name}"
        group_dir_path = os.path.join(OUTPUT_DIR, group_dir_name)
        os.makedirs(group_dir_path, exist_ok=True)
        
        # Store mapping for later use
        group_dirs[base_model] = group_dir_path
        
        # Create directories for each model in the group
        for model in models:
            model_dir = os.path.join(group_dir_path, get_model_dir(model))
            os.makedirs(model_dir, exist_ok=True)
    
    return group_dirs

def find_task_jsonl_files(group_dir, model_dir, task_name):
    """Find all JSONL files for a specific task in the model directory"""
    import glob
    
    # The full path to the model directory
    full_model_dir = os.path.join(group_dir, model_dir)
    matching_files = []
    
    # First, look in the samples directory
    samples_dir = os.path.join(full_model_dir, "samples")
    if os.path.exists(samples_dir):
        print(samples_dir)
        pattern = os.path.join(samples_dir, f"samples_{task_name}_*.jsonl")
        matching_files.extend(glob.glob(pattern))

    if not matching_files:
        print(f"Warning: No JSONL files found for {task_name} in {full_model_dir}")
        
    return matching_files

def extract_accuracy_from_results(results):
    """Extract the main accuracy metric from results dictionary"""
    # Check common accuracy metrics in order of preference
    if "acc,none" in results:
        return results["acc,none"]
    elif "acc_norm,none" in results:
        return results["acc_norm,none"]
    elif "exact_match,strict-match" in results:
        return results["exact_match,strict-match"]
    elif "mean_acc" in results:
        return results["mean_acc"]
    
    # If we can't find a direct accuracy metric, look for any key containing "acc"
    for key in results:
        if "acc" in key.lower():
            return results[key]
    
    # Default to None if no accuracy measure found
    return None

def is_vllm_quantized_model(model_name):
    """Check if model name contains quantization pattern like w8a8 that requires vllm."""
    # Look for patterns like w4a16, w8a8, etc.
    return bool(re.search(r'w\d+a\d+', model_name.lower()))

def run_task_evaluations_and_flips(config, model_groups, group_dirs):
    """Run evaluations for a specific task config and calculate flips between model pairs"""
    task_name = config["task_name"]
    print(f"\n{'='*60}\nEvaluating task: {task_name}\n{'='*60}")
    
    # Dictionary to store results for each model
    task_results = {}
    # Dictionary to track which models use vllm (for KL divergence calculation)
    vllm_models = {}
    
    # First pass: Run evaluations on all models
    for models in model_groups:
        base_model = models[0]
        group_dir = group_dirs[base_model]
        
        # Store results for this model group
        group_results = {}
        
        for model_name in models:
            # Check if this is a vLLM quantized model
            is_vllm_model = is_vllm_quantized_model(model_name)
            vllm_models[model_name] = is_vllm_model
            
            # Configure model args based on model type
            if is_vllm_model:
                print(f"\nDetected quantized model requiring vLLM: {model_name}")
                config["model"] = "vllm"
                config["model_args"] = f"pretrained={model_name},max_model_len=4096,gpu_memory_utilization=0.8,tensor_parallel_size=1"
                config["batch_size"] = "auto"
                # Clean up any previous vLLM processes
                cleanup_vllm_processes()
            else:
                config["model"] = "hf"
            config["model_args"] = f"pretrained={model_name}"
            
            config["limit"] = None  # Set to None for full evaluation or some number for testing
            
            # Get the model directory (without samples subdirectory)
            model_dir = os.path.join(group_dir, get_model_dir(model_name))
            samples_dir = os.path.join(model_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            print(f"\nEvaluating model: {model_name} on {task_name}")
            print(f"Using engine: {config['model']}")
            clear_memory()
            
            # Initialize the tracker with the model-specific directory and model name
            eval_tracker = EvaluationTracker(output_path=model_dir)
            # Make sure model_name is set in the tracker's config
            eval_tracker.general_config_tracker.model_name_sanitized = model_name
            eval_tracker.general_config_tracker.model_name_sanitized = "samples"
            eval_tracker.date_id = datetime.now().isoformat().replace(":", "-")
            
            # Pass the model directory as output_path for storing logits
            results = lm_eval.simple_evaluate(
                model=config["model"],
                model_args=config["model_args"],
                tasks=[config["task_name"]],
                num_fewshot=config.get("num_fewshot", 0),
                batch_size=config.get("batch_size", None),
                limit=config.get("limit", None),
                log_samples=True,
                output_path=model_dir
            )
            md = lm_eval.utils.make_table(results)
            
            # Extract and save samples if they exist
            if "samples" in results:
                samples = results.pop("samples")
                # Create samples directory
                samples_dir = os.path.join(model_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                # Save samples for each task using the tracker only
                for task_name, task_samples in samples.items():
                    # Use the tracker to save samples
                    eval_tracker.save_results_samples(
                        task_name=task_name,
                        samples=task_samples
                    )
                    print(f"Samples for {task_name} saved via tracker to {model_dir}")
            
            # Extract the raw results data
            results_dict = results["results"][task_name]
            accuracy = extract_accuracy_from_results(results_dict)
            
            # Save results to model directory
            results_filename = os.path.join(model_dir, f"{task_name}_results.md")
            with open(results_filename, "w") as f:
                f.write(md)
            
            # Also save raw results as JSON for easier processing
            results_json_filename = os.path.join(model_dir, f"{task_name}_results.jsonl")
            with open(results_json_filename, "w") as f:
                json.dump(results_dict, f, indent=2)
                
            # Store results for recovery calculation
            group_results[model_name] = {
                "accuracy": accuracy,
                "results": results_dict
            }
            
            print(f"Results saved to {results_filename}")
        
        # Store the group results
        task_results[base_model] = group_results
    
    # Second pass: Run flip evaluation and calculate recovery percentage for each model group
    print(f"\nRunning flip evaluations for {task_name}")
    
    for models in model_groups:
        base_model, *other_models = models
        base_accuracy = task_results[base_model][base_model]["accuracy"]
        group_dir = group_dirs[base_model]
        
        # Prepare aggregated metrics for this task and model group
        aggregated_metrics = {
            "task_name": task_name,
            "base_model": base_model,
            "base_accuracy": base_accuracy,
            "quantized_models": []
        }
        
        # Add KL divergence results storage
        kl_metrics = {
            "task_name": task_name,
            "base_model": base_model,
            "quantized_models": []
        }
        
        for new_model in other_models:
            base_model_dir = get_model_dir(base_model)
            new_model_dir = get_model_dir(new_model)
            
            # Check if either model uses vLLM (for KL divergence calculation)
            base_uses_vllm = vllm_models.get(base_model, False)
            new_uses_vllm = vllm_models.get(new_model, False)
            skip_kl = base_uses_vllm or new_uses_vllm
            
            # Find the JSONL files for this task
            base_jsonl_files = find_task_jsonl_files(group_dir, base_model_dir, task_name)
            new_jsonl_files = find_task_jsonl_files(group_dir, new_model_dir, task_name)
            
            if base_jsonl_files and new_jsonl_files:
                print(f"\nComparing {base_model} vs {new_model} on {task_name}")
                print(f"Found {len(base_jsonl_files)} base files and {len(new_jsonl_files)} new files")
                
                # Initialize FlipEval with all files for this task/subtasks
                flip_evaluator = FlipEval(
                    base_path=os.path.join(group_dir, base_model_dir),
                    new_path=os.path.join(group_dir, new_model_dir),
                    task_name=task_name
                )
                
                # Run the evaluation across all matching files
                stats = flip_evaluator.analyze_all_task_files(base_jsonl_files, new_jsonl_files, verbose=True)
                
                # Calculate recovery percentage
                new_accuracy = task_results[base_model][new_model]["accuracy"]
                recovery_pct = (new_accuracy / base_accuracy * 100) if base_accuracy else 0.0
                
                # Save detailed results
                model_comparison_dir = os.path.join(group_dir, "comparisons") 
                os.makedirs(model_comparison_dir, exist_ok=True)
                
                flip_results_filename = os.path.join(model_comparison_dir, f"{task_name}_{base_model_dir}_vs_{new_model_dir}.txt")
                
                with open(flip_results_filename, "w") as f:
                    f.write(f"FLIP ANALYSIS: {base_model} vs {new_model} on {task_name}\n")
                    print(base_accuracy)
                    f.write(f"Base Accuracy: {base_accuracy:.4f}\n")
                    f.write(f"New Accuracy: {new_accuracy:.4f}\n")
                    f.write(f"Recovery: {recovery_pct:.2f}%\n")
                    f.write(f"Total examples: {stats['total_examples']}\n")
                    f.write(f"Total flips: {stats['total_flips']} ({stats['percent_flips']:.2f}%)\n")
                    f.write(f"  Correct → Wrong: {stats['correct_to_wrong']}\n")
                    f.write(f"  Wrong → Correct: {stats['wrong_to_correct']}\n")
                    
                    # Include per-subtask details
                    f.write("\nPER SUBTASK ANALYSIS:\n")
                    for subtask, data in stats['per_subtask'].items():
                        f.write(f"{subtask:<30} Total: {data['total']:<7} Flips: {data['total_flips']:<7} ")
                        f.write(f"C→W: {data['correct_to_wrong']:<7} W→C: {data['wrong_to_correct']:<7} ")
                        f.write(f"Flip %: {data['percent_flips']:.2f}%\n")
                
                # Store for aggregated metrics
                aggregated_metrics["quantized_models"].append({
                    "model": new_model,
                    "accuracy": new_accuracy,
                    "recovery_pct": recovery_pct,
                    "flips": {
                        "total": stats['total_flips'],
                        "percent": stats['percent_flips'],
                        "correct_to_wrong": stats['correct_to_wrong'],
                        "wrong_to_correct": stats['wrong_to_correct']
                    },
                    "subtasks": stats['per_subtask']
                })
                
                print(f"Detailed flip results saved to {flip_results_filename}")
                print(f"Recovery percentage: {recovery_pct:.2f}%")
                
                # Run KL evaluation
                # Get the paths to the tensor files directories
                base_model_path = os.path.join(group_dir, base_model_dir)
                new_model_path = os.path.join(group_dir, new_model_dir)
                
                if skip_kl:
                    print(f"\nSkipping KL divergence calculation because one or both models use vLLM")
                    print(f"  Base model uses vLLM: {base_uses_vllm}")
                    print(f"  New model uses vLLM: {new_uses_vllm}")
                else:
                    print(f"\nCalculating KL divergence between {base_model} and {new_model}")
                    try:
                        # Initialize KL evaluator
                        kl_evaluator = KLEval(base_model_path, new_model_path)
                        
                        # Calculate KL divergence
                        kl_output_path = os.path.join(model_comparison_dir, f"{task_name}_{base_model_dir}_vs_{new_model_dir}_kl.pt")
                        kl_results = kl_evaluator.save_kl_results(kl_output_path)
                        
                        # Store KL results in metrics
                        task_kl = kl_results.get(task_name, {}).get("mean_kl", None)
                        if task_kl is not None:
                            kl_metrics["quantized_models"].append({
                                "model": new_model,
                                "kl_divergence": task_kl,
                                "full_results": kl_results
                            })
                            print(f"KL divergence: {task_kl:.6f}")
                        else:
                            print(f"Warning: KL divergence for task {task_name} not available")
                    except Exception as e:
                        print(f"Error calculating KL divergence: {e}")
                        raise e
            else:
                print(f"ERROR: Could not find JSONL files for task {task_name} with models {models}")
                if not base_jsonl_files:
                    print(f"Missing base model JSONL files in {base_model_dir}")
                if not new_jsonl_files:
                    print(f"Missing new model JSONL files in {new_model_dir}")
        
        # Save aggregated metrics to model group directory
        aggregated_filename = os.path.join(group_dir, f"{task_name}_aggregated_metrics.json")
        with open(aggregated_filename, "w") as f:
            json.dump(aggregated_metrics, f, indent=2)
        
        # Save KL metrics if available
        if any(kl_metrics["quantized_models"]):
            kl_filename = os.path.join(group_dir, f"{task_name}_kl_metrics.json")
            with open(kl_filename, "w") as f:
                # Filter out full_results from the JSON to keep it clean
                clean_kl_metrics = {
                    "task_name": kl_metrics["task_name"],
                    "base_model": kl_metrics["base_model"],
                    "quantized_models": [
                        {k: v for k, v in model.items() if k != "full_results"}
                        for model in kl_metrics["quantized_models"]
                    ]
                }
                json.dump(clean_kl_metrics, f, indent=2)
        
        # Also create a readable markdown summary with both metrics
        markdown_summary = os.path.join(group_dir, f"{task_name}_comparison.md")
        with open(markdown_summary, "w") as f:
            f.write(f"# {task_name.upper()} Model Group Comparison\n\n")
            f.write(f"Base Model: **{base_model}**\n\n")
            f.write(f"Base Accuracy: **{base_accuracy:.4f}**\n\n")
            f.write("| Model | Accuracy | Recovery % | Flips % | Correct→Wrong | Wrong→Correct | KL Divergence |\n")
            f.write("|-------|----------|-----------|---------|---------------|---------------|---------------|\n")
            
            # Add row for base model
            f.write(f"| {base_model} | {base_accuracy:.4f} | 100.00% | - | - | - | - |\n")
            
            # Add rows for quantized models
            for model_data in aggregated_metrics["quantized_models"]:
                # Find KL value for this model if available
                kl_value = "-"
                for kl_model in kl_metrics.get("quantized_models", []):
                    if kl_model["model"] == model_data["model"]:
                        kl_value = f"{kl_model['kl_divergence']:.6f}"
                        break
                
                # Mark if using vLLM (which doesn't support KL)
                model_name = model_data["model"]
                if vllm_models.get(model_name, False):
                    kl_value = "N/A (vLLM)"
                
                f.write(f"| {model_data['model']} | {model_data['accuracy']:.4f} | {model_data['recovery_pct']:.2f}% | ")
                f.write(f"{model_data['flips']['percent']:.2f}% | {model_data['flips']['correct_to_wrong']} | ")
                f.write(f"{model_data['flips']['wrong_to_correct']} | {kl_value} |\n")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # Create directory structure first
    print("Creating directory structure...")
    group_dirs = create_directory_structure(model_groups)
    
    # Run evaluations for each config
    configs = [hellaswag_config, mmlu_config, arc_c_config, 
              gsm8kcot_config, winogrande_config, truthfulqa_config]
    
    # # For testing, you may want to run just one config
    #configs = [hellaswag_config]
    
    # Create a metadata file with run information
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_groups": model_groups,
        "configs": [config["task_name"] for config in configs]
    }
    
    with open(os.path.join(OUTPUT_DIR, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    for config in configs:
        run_task_evaluations_and_flips(config, model_groups, group_dirs)
    
    # Clean up any remaining vLLM processes
    cleanup_vllm_processes()
else:
    # When imported as a module, initialize multiprocessing appropriately
    # This is important for vLLM's process management
    pass

