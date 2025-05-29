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
