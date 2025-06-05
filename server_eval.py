import lm_eval
import lm_eval.utils
import os
import json
import gc
import torch
import argparse
from datetime import datetime
from lm_eval.loggers import EvaluationTracker

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


def clear_memory():
    """Clear memory to prevent OOM issues between evaluations"""
    # Clear all unused PyTorch memory caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    # Memory info
    if torch.cuda.is_available():
        print(
            f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved"
        )


def get_model_dir(model_name):
    """Convert model name to directory name format"""
    return model_name.replace("/", "__")


def evaluate_server_model(
    config: dict,
    model_name: str,
    base_url: str,
    output_dir: str = "server_eval_output",
    num_concurrent: int = 1,
    max_retries: int = 3,
    tokenized_requests: bool = False,
):
    """
    Evaluate a model served via API endpoint using lm_eval.

    Args:
        config: Configuration dict with task_name, batch_size, etc.
        model_name: Name of the model (e.g., "facebook/opt-125m")
        base_url: Server endpoint URL (e.g., "http://localhost:8000/v1/completions")
        output_dir: Directory to save results
        num_concurrent: Number of concurrent requests
        max_retries: Number of retry attempts
        tokenized_requests: Whether to use tokenized requests
    """

    task_name = config["task_name"]
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Task: {task_name}")
    print(f"Server URL: {base_url}")
    print(f"{'='*60}")

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, get_model_dir(model_name))
    os.makedirs(model_dir, exist_ok=True)
    samples_dir = os.path.join(model_dir, "samples")
    print(samples_dir)
    os.makedirs(samples_dir, exist_ok=True)

    # Configure model arguments for server evaluation
    model_args = (
        f"model={model_name},"
        f"base_url={base_url},"
        f"num_concurrent={num_concurrent},"
        f"max_retries={max_retries},"
        f"tokenized_requests={tokenized_requests},"
        f"batch_size={config.get('batch_size', 16)}"
    )

    print(f"Model args: {model_args}")

    # Clear memory before evaluation
    clear_memory()

    # Initialize the evaluation tracker
    eval_tracker = EvaluationTracker(output_path=model_dir)
    eval_tracker.general_config_tracker.model_name_sanitized = "samples"
    eval_tracker.date_id = datetime.now().isoformat().replace(":", "-")

    try:
        # Run the evaluation
        print(f"\nStarting evaluation...")
        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=[task_name],
            limit=config.get("limit", None),
            log_samples=config.get("log_samples", True),
            num_fewshot=config.get("num_fewshot", 0),
        )

        print(f"✅ Evaluation completed successfully!")

        # Generate markdown table of results
        md = lm_eval.utils.make_table(results)

        # Extract and save samples if they exist
        if "samples" in results:
            samples = results.pop("samples")
            for task_name_key, task_samples in samples.items():
                eval_tracker.save_results_samples(
                    task_name=task_name_key, samples=task_samples
                )
                print(f"📁 Samples for {task_name_key} saved to {model_dir}")

        # Extract the results for this task
        results_dict = results["results"][task_name]

        # Save results as markdown
        results_filename = os.path.join(model_dir, f"{task_name}_results.md")
        with open(results_filename, "w") as f:
            f.write(md)
        print(f"📄 Results saved to {results_filename}")

        # Save raw results as JSON
        results_json_filename = os.path.join(model_dir, f"{task_name}_results.json")
        with open(results_json_filename, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"📊 Raw results saved to {results_json_filename}")

        # Save full results with metadata
        full_results_filename = os.path.join(
            model_dir, f"{task_name}_full_results.json"
        )
        with open(full_results_filename, "w") as f:
            json.dump(results, f, indent=2)

        # Extract main accuracy metric
        accuracy = extract_accuracy_from_results(results_dict)
        if accuracy is not None:
            print(f"🎯 Main accuracy metric: {accuracy:.4f}")

        # Create summary
        summary = {
            "model_name": model_name,
            "task_name": task_name,
            "server_url": base_url,
            "accuracy": accuracy,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "results": results_dict,
        }

        summary_filename = os.path.join(model_dir, f"{task_name}_summary.json")
        with open(summary_filename, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Summary saved to {summary_filename}")

        return results

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise


def extract_accuracy_from_results(results):
    """Extract the main accuracy metric from results dictionary"""
    # Check common accuracy metrics in order of preference
    if "acc,none" in results:
        return results["acc,none"]
    elif "acc_norm,none" in results:
        return results["acc_norm,none"]
    elif "exact_match,strict-match" in results:
        return results["exact_match,strict-match"]
    elif "exact_match,flexible-extract" in results:
        return results["exact_match,flexible-extract"]
    elif "mean_acc" in results:
        return results["mean_acc"]

    # If we can't find a direct accuracy metric, look for any key containing "acc"
    for key in results:
        if "acc" in key.lower():
            return results[key]

    # Default to None if no accuracy measure found
    return None


def evaluate_multiple_tasks(
    model_name: str,
    base_url: str,
    configs: list,
    output_dir: str = "server_eval_output",
):
    """
    Evaluate a server model on multiple tasks.

    Args:
        model_name: Name of the model
        base_url: Server endpoint URL
        configs: List of configuration dictionaries
        output_dir: Output directory for results
    """
    results = {}

    for config in configs:
        print(f"\n🚀 Running evaluation for task: {config['task_name']}")
        try:
            task_results = evaluate_server_model(
                config=config,
                model_name=model_name,
                base_url=base_url,
                output_dir=output_dir,
            )
            results[config["task_name"]] = task_results

        except Exception as e:
            print(f"❌ Failed to evaluate task {config['task_name']}: {e}")
            results[config["task_name"]] = None

    # Save combined results
    combined_filename = os.path.join(output_dir, "combined_results.json")
    with open(combined_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Combined results saved to {combined_filename}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model served via API endpoint using lm_eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and server configuration
    parser.add_argument(
        "--model-name", 
        type=str, 
        required=True,
        help="Name of the model (e.g., 'facebook/opt-125m')"
    )
    parser.add_argument(
        "--base-url", 
        type=str, 
        default="http://0.0.0.0:8000/v1/completions",
        help="Server endpoint URL"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="server_eval_output",
        help="Directory to save results"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--task-name", 
        type=str, 
        default="hellaswag",
        help="Task name for evaluation (e.g., 'hellaswag', 'openllm', 'gsm8k')"
    )
    parser.add_argument(
        "--log-samples", 
        action="store_true", 
        default=True,
        help="Whether to log samples"
    )
    parser.add_argument(
        "--no-log-samples", 
        action="store_false", 
        dest="log_samples",
        help="Disable logging samples"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-fewshot", 
        type=int, 
        default=0,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of samples for testing (None for no limit)"
    )
    
    # Server request configuration
    parser.add_argument(
        "--num-concurrent", 
        type=int, 
        default=1,
        help="Number of concurrent requests"
    )
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3,
        help="Number of retry attempts"
    )
    parser.add_argument(
        "--tokenized-requests", 
        action="store_true", 
        default=False,
        help="Whether to use tokenized requests"
    )
    
    args = parser.parse_args()
    
    # Create configuration dict from arguments
    config = {
        "task_name": args.task_name,
        "log_samples": args.log_samples,
        "batch_size": args.batch_size,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
    }
    
    print("🔧 Running evaluation with configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Task: {args.task_name}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Few-shot: {args.num_fewshot}")
    print(f"  Limit: {args.limit}")
    print(f"  Log samples: {args.log_samples}")
    
    # Run evaluation
    try:
        results = evaluate_server_model(
            config=config,
            model_name=args.model_name,
            base_url=args.base_url,
            output_dir=args.output_dir,
            num_concurrent=args.num_concurrent,
            max_retries=args.max_retries,
            tokenized_requests=args.tokenized_requests,
        )
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        exit(1)
