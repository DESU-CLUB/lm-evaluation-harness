import json
import os
from full_eval import (
    find_task_jsonl_files,
    get_model_dir,
    extract_base_model_name,
    is_vllm_quantized_model,
)
from flip_eval import FlipEval
from KLEval import KLEval
import re
from typing import Dict, Any, List


def _run_flip_analysis(
    config,
    model_groups,
    group_dirs,
    task_results,
    vllm_models,
    kl_divergence=False,
    PATH=None,
):
    """Run flip evaluation and KL analysis (shared between both approaches)"""
    task_name = config["task_name"]

    metric_keys = {
        "mmlu": "acc,none",
        "hellaswag": "acc,none",
        "winogrande": "acc,none",
        "truthfulqa_mc2": "acc,none",
        "arc_challenge": "acc,none",
        "gsm8k": "exact_match,flexible-extract",
    }

    # Second pass: Run flip evaluation and calculate recovery percentage for each model group
    print(f"\nRunning flip evaluations for {task_name}")

    for idx, models in enumerate(model_groups):
        base_model, *other_models = models
        print(task_name, metric_keys[task_name])
        base_accuracy = task_results[base_model][base_model]["results"][task_name][
            metric_keys[task_name]
        ]
        group_dir = group_dirs[base_model]

        # Prepare aggregated metrics for this task and model group
        aggregated_metrics = {
            "task_name": task_name,
            "base_model": base_model,
            "base_accuracy": base_accuracy,
            "quantized_models": [],
        }

        # Add KL divergence results storage
        kl_metrics = {
            "task_name": task_name,
            "base_model": base_model,
            "quantized_models": [],
        }
        group_dir = os.path.join(PATH[idx], group_dirs[base_model])
        for new_model in other_models:
            base_model_dir = get_model_dir(base_model)
            print(base_model_dir)
            new_model_dir = get_model_dir(new_model)

            # Check if either model uses vLLM (for KL divergence calculation)
            base_uses_vllm = vllm_models.get(base_model, False)
            new_uses_vllm = vllm_models.get(new_model, False)
            skip_kl = base_uses_vllm or new_uses_vllm or not kl_divergence

            # Find the JSONL files for this task
            base_jsonl_files = find_task_jsonl_files(
                group_dir, base_model_dir, task_name
            )
            new_jsonl_files = find_task_jsonl_files(group_dir, new_model_dir, task_name)

            # Ensure deterministic order so files correspond 1-to-1
            base_jsonl_files.sort()
            new_jsonl_files.sort()

            if base_jsonl_files and new_jsonl_files:
                print(f"\nComparing {base_model} vs {new_model} on {task_name}")
                print(
                    f"Found {len(base_jsonl_files)} base files and {len(new_jsonl_files)} new files"
                )

                # Initialize FlipEval with all files for this task/subtasks
                flip_evaluator = FlipEval(
                    base_path=os.path.join(group_dir, base_model_dir),
                    new_path=os.path.join(group_dir, new_model_dir),
                    task_name=task_name,
                )

                # Run the evaluation across all matching files
                stats = flip_evaluator.analyze_all_task_files(
                    base_jsonl_files, new_jsonl_files, verbose=True
                )

                # Calculate recovery percentage
                new_accuracy = task_results[base_model][new_model]["results"][
                    task_name
                ][metric_keys[task_name]]
                recovery_pct = (
                    (new_accuracy / base_accuracy * 100) if base_accuracy else 0.0
                )

                # Save detailed results
                model_comparison_dir = os.path.join(group_dir, "comparisons")
                os.makedirs(model_comparison_dir, exist_ok=True)

                flip_results_filename = os.path.join(
                    model_comparison_dir,
                    f"{task_name}_{base_model_dir}_vs_{new_model_dir}.txt",
                )

                with open(flip_results_filename, "w") as f:
                    f.write(
                        f"FLIP ANALYSIS: {base_model} vs {new_model} on {task_name}\n"
                    )
                    f.write(f"Base Accuracy: {base_accuracy:.4f}\n")
                    f.write(f"New Accuracy: {new_accuracy:.4f}\n")
                    f.write(f"Recovery: {recovery_pct:.2f}%\n")
                    f.write(f"Total examples: {stats['total_examples']}\n")
                    f.write(
                        f"Total flips: {stats['total_flips']} ({stats['percent_flips']:.2f}%)\n"
                    )
                    f.write(f"  Correct → Wrong: {stats['correct_to_wrong']}\n")
                    f.write(f"  Wrong → Correct: {stats['wrong_to_correct']}\n")

                    # Include per-subtask details
                    f.write("\nPER SUBTASK ANALYSIS:\n")
                    for subtask, data in stats["per_subtask"].items():
                        f.write(
                            f"{subtask:<30} Total: {data['total']:<7} Flips: {data['total_flips']:<7} "
                        )
                        f.write(
                            f"C→W: {data['correct_to_wrong']:<7} W→C: {data['wrong_to_correct']:<7} "
                        )
                        f.write(f"Flip %: {data['percent_flips']:.2f}%\n")

                # Store for aggregated metrics
                aggregated_metrics["quantized_models"].append(
                    {
                        "model": new_model,
                        "accuracy": new_accuracy,
                        "recovery_pct": recovery_pct,
                        "flips": {
                            "total": stats["total_flips"],
                            "percent": stats["percent_flips"],
                            "correct_to_wrong": stats["correct_to_wrong"],
                            "wrong_to_correct": stats["wrong_to_correct"],
                        },
                        "subtasks": stats["per_subtask"],
                    }
                )

                print(f"Detailed flip results saved to {flip_results_filename}")
                print(f"Recovery percentage: {recovery_pct:.2f}%")

                # Run KL evaluation (only if enabled and not skipped)
                if not skip_kl:
                    print(
                        f"\nCalculating KL divergence between {base_model} and {new_model}"
                    )
                    base_model_path = os.path.join(group_dir, base_model_dir)
                    new_model_path = os.path.join(group_dir, new_model_dir)

                    try:
                        # Initialize KL evaluator
                        kl_evaluator = KLEval(base_model_path, new_model_path)

                        # Calculate KL divergence
                        kl_output_path = os.path.join(
                            model_comparison_dir,
                            f"{task_name}_{base_model_dir}_vs_{new_model_dir}_kl.pt",
                        )
                        kl_results = kl_evaluator.save_kl_results(kl_output_path)

                        # Store KL results in metrics
                        task_kl = kl_results.get(task_name, {}).get("mean_kl", None)
                        if task_kl is not None:
                            kl_metrics["quantized_models"].append(
                                {
                                    "model": new_model,
                                    "kl_divergence": task_kl,
                                    "full_results": kl_results,
                                }
                            )
                            print(f"KL divergence: {task_kl:.6f}")
                        else:
                            print(
                                f"Warning: KL divergence for task {task_name} not available"
                            )
                    except Exception as e:
                        print(f"Error calculating KL divergence: {e}")
                elif skip_kl:
                    if not kl_divergence:
                        print(
                            f"\nSkipping KL divergence calculation because kl_divergence flag is False"
                        )
                    else:
                        print(
                            f"\nSkipping KL divergence calculation because one or both models use vLLM"
                        )
                        print(f"  Base model uses vLLM: {base_uses_vllm}")
                        print(f"  New model uses vLLM: {new_uses_vllm}")
            else:
                print(
                    f"ERROR: Could not find JSONL files for task {task_name} with models {models}"
                )
                if not base_jsonl_files:
                    print(f"Missing base model JSONL files in {base_model_dir}")
                if not new_jsonl_files:
                    print(f"Missing new model JSONL files in {new_model_dir}")

        # Save aggregated metrics and create markdown summary (same as before)
        # ... (rest of the function remains the same)

        # Save aggregated metrics to model group directory
        aggregated_filename = os.path.join(
            group_dir, f"{task_name}_aggregated_metrics.json"
        )
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
                    ],
                }
                json.dump(clean_kl_metrics, f, indent=2)

        # Also create a readable markdown summary with both metrics
        markdown_summary = os.path.join(group_dir, f"{task_name}_comparison.md")
        with open(markdown_summary, "w") as f:
            f.write(f"# {task_name.upper()} Model Group Comparison\n\n")
            f.write(f"Base Model: **{base_model}**\n\n")
            f.write(f"Base Accuracy: **{base_accuracy:.4f}**\n\n")
            f.write(
                "| Model | Accuracy | Recovery % | Flips % | Correct→Wrong | Wrong→Correct | KL Divergence |\n"
            )
            f.write(
                "|-------|----------|-----------|---------|---------------|---------------|---------------|\n"
            )

            # Add row for base model
            f.write(
                f"| {base_model} | {base_accuracy:.4f} | 100.00% | - | - | - | - |\n"
            )

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

                f.write(
                    f"| {model_data['model']} | {model_data['accuracy']:.4f} | {model_data['recovery_pct']:.2f}% | "
                )
                f.write(
                    f"{model_data['flips']['percent']:.2f}% | {model_data['flips']['correct_to_wrong']} | "
                )
                f.write(f"{model_data['flips']['wrong_to_correct']} | {kl_value} |\n")



# ---------------------------------------------------------------------------
# Utility:  parse the comparison markdown back into JSON
# ---------------------------------------------------------------------------


def comparison_md_to_json(md_path: str, save: bool = True) -> Dict[str, Any]:
    """Parse a *_comparison.md file generated by ``_run_flip_analysis``.

    The markdown table section looks like::

        | Model | Accuracy | Recovery % | Flips % | Correct→Wrong | Wrong→Correct | KL Divergence |
        |-------|----------|-----------|---------|---------------|---------------|---------------|
        | base_model | 0.8321 | 100.00% | - | - | - | - |
        | quant_model | 0.8012 | 96.29% |  4.27% | 30 | 42 | 0.004512 |

    This helper extracts each row into a dict and returns a mapping::

        {
            "base_model_name": {...},
            "quant_model_name": {...}
        }

    If *save* is True (default) a ``<md_path_without_ext>.json`` file is
    written alongside the markdown.
    """

    md_path = os.path.abspath(md_path)
    if not os.path.exists(md_path):
        raise FileNotFoundError(md_path)

    data_rows: List[Dict[str, Any]] = []
    inside_table = False

    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Detect the start of the table header
            if line.startswith("| Model |"):
                inside_table = True
                continue  # skip header row

            if inside_table:
                # Skip the delimiter row made of hyphens
                if re.match(r"\|[- ]+\|", line):
                    continue

                # End of table: blank line or something not starting with '|'
                if not line.startswith("|"):
                    break

                # Split columns, strip whitespace
                cols = [c.strip() for c in line.strip("|").split("|")]
                if len(cols) < 7:
                    continue  # malformed row; ignore

                try:
                    accuracy = float(cols[1]) if cols[1] not in ("-", "") else None
                    recovery = (
                        float(cols[2].replace("%", "")) if "%" in cols[2] else None
                    )
                    flips_pct = (
                        float(cols[3].replace("%", "")) if "%" in cols[3] else None
                    )
                    correct_to_wrong = int(cols[4]) if cols[4].isdigit() else None
                    wrong_to_correct = int(cols[5]) if cols[5].isdigit() else None
                    kl_value = None
                    if cols[6] not in ("-", "N/A (vLLM)"):
                        try:
                            kl_value = float(cols[6])
                        except ValueError:
                            kl_value = cols[6]
                except Exception:
                    # fallback: store raw strings if parsing fails
                    (
                        accuracy,
                        recovery,
                        flips_pct,
                        correct_to_wrong,
                        wrong_to_correct,
                        kl_value,
                    ) = [None] * 6

                data_rows.append(
                    {
                        "model": cols[0],
                        "accuracy": accuracy,
                        "recovery_pct": recovery,
                        "flips_pct": flips_pct,
                        "correct_to_wrong": correct_to_wrong,
                        "wrong_to_correct": wrong_to_correct,
                        "kl_divergence": kl_value,
                    }
                )

    # Convert to mapping keyed by model name for convenience
    json_data = {row["model"]: row for row in data_rows}

    if save:
        json_path = os.path.splitext(md_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)
        print(f"[INFO] comparison_md_to_json → saved {json_path}")

    return json_data


# ---------------------------------------------------------------------------
# Utility: parse a *results.md table (generated by lm_eval.utils.make_table)
#          back into the original per-task JSON structure that simple_evaluate
#          would have produced.  This lets us recreate the data when only the
#          markdown is available.
# ---------------------------------------------------------------------------


def results_md_to_json(md_path: str, save: bool = False) -> Dict[str, Any]:
    """Parse `<task>_results.md` produced by `make_table()`.

    Returns a dict of the form expected from `simple_evaluate`, i.e.

    ```python
    {
        "results": {
            "arc_challenge": {"acc,none": 0.58, "acc_norm,none": 0.62},
            "hellaswag": {"acc,none": 0.589, ...},
            ...
        }
    }
    ```
    """

    md_path = os.path.abspath(md_path)
    if not os.path.exists(md_path):
        raise FileNotFoundError(md_path)

    # Prepare containers
    task_results: Dict[str, Dict[str, Any]] = {}

    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # we only care about table rows (start with '|') and that are not header / delimiter
            if not line.startswith("|"):
                continue

            if re.match(r"\|[- ]+\|", line):  # delimiter row of dashes
                continue

            # Split cells
            cells = [c.strip() for c in line.strip("|").split("|")]

            # Header line contains "Tasks" etc – skip
            if cells and cells[0] == "Tasks":
                continue

            # Expect at least 7 columns (as produced by make_table)
            if len(cells) < 7:
                continue

            raw_task = cells[0]
            # Ignore the summary row "Open LLM Leaderboard" or empty task cells
            if raw_task == "" or "Leaderboard" in raw_task:
                continue

            # Clean task field: strip any leading dashes / bullets and excess spaces
            task_name = raw_task.lstrip("- ").strip()

            metric = cells[4]
            value_str = cells[6]
            filter_name = cells[2] or "none"

            # Some rows correspond to stderr – skip those (handled in make_table)
            if metric.endswith("_stderr"):
                continue

            # Convert value to float when possible
            try:
                value = float(value_str)
            except ValueError:
                continue

            key = f"{metric},{filter_name}"
            task_results.setdefault(task_name, {})[key] = value

    json_dict: Dict[str, Any] = {"results": task_results}

    if save:
        json_path = os.path.splitext(md_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_dict, jf, indent=2)
        print(f"[INFO] results_md_to_json → saved {json_path}")

    return json_dict


# ---------------------------------------------------------------------------
# Utility: consolidate all task markdown files into a single summary
# ---------------------------------------------------------------------------


def consolidate_task_markdowns(model_groups, group_dirs, PATH, tasks=None):
    """Consolidate all task comparison markdown files into a single comprehensive report.
    
    Args:
        model_groups: List of model group tuples
        group_dirs: Dictionary mapping base models to their group directory names
        PATH: List of base paths for each model group
        tasks: List of task names to consolidate (if None, uses default OpenLLM tasks)
    """
    if tasks is None:
        tasks = ["mmlu", "hellaswag", "winogrande", "truthfulqa_mc2", "arc_challenge", "gsm8k"]
    
    for idx, models in enumerate(model_groups):
        base_model = models[0]
        group_dir = os.path.join(PATH[idx], group_dirs[base_model])
        
        # Create consolidated markdown file
        consolidated_path = os.path.join(group_dir, "consolidated_comparison.md")
        
        # Collect data for leaderboard table
        leaderboard_data = {}
        
        with open(consolidated_path, "w") as consolidated_f:
            # Write header
            consolidated_f.write(f"# Comprehensive Model Group Comparison\n\n")
            consolidated_f.write(f"**Base Model:** {base_model}\n\n")
            consolidated_f.write(f"**Quantized Models:** {', '.join(models[1:])}\n\n")
            consolidated_f.write("---\n\n")
            
            # Process each task
            for task_idx, task in enumerate(tasks):
                task_md_path = os.path.join(group_dir, f"{task}_comparison.md")
                
                if os.path.exists(task_md_path):
                    consolidated_f.write(f"## {task.upper()} Results\n\n")
                    
                    # Read and append the task markdown content
                    with open(task_md_path, "r") as task_f:
                        content = task_f.read()
                        
                        # Skip the first line (main header) and add the content
                        lines = content.split('\n')
                        # Find where the actual content starts (after the main header)
                        content_start = 0
                        for i, line in enumerate(lines):
                            if line.startswith("Base Model:"):
                                content_start = i
                                break
                        
                        # Write the content from base model info onwards
                        consolidated_f.write('\n'.join(lines[content_start:]))
                    
                    # Parse the task comparison file to extract data for leaderboard
                    task_comparison_path = os.path.join(group_dir, f"{task}_comparison.json")
                    if os.path.exists(task_comparison_path):
                        # Try to load from JSON first
                        try:
                            with open(task_comparison_path, "r") as json_f:
                                task_data = json.load(json_f)
                                for model_name, model_info in task_data.items():
                                    if model_name not in leaderboard_data:
                                        leaderboard_data[model_name] = {}
                                    leaderboard_data[model_name][task] = model_info
                        except (json.JSONDecodeError, FileNotFoundError):
                            pass
                    
                    # If JSON doesn't exist, parse from markdown
                    if task not in leaderboard_data.get(base_model, {}):
                        task_data = comparison_md_to_json(task_md_path, save=False)
                        for model_name, model_info in task_data.items():
                            if model_name not in leaderboard_data:
                                leaderboard_data[model_name] = {}
                            leaderboard_data[model_name][task] = model_info
                    
                    # Add separator between tasks (except for the last one)
                    if task_idx < len(tasks) - 1:
                        consolidated_f.write("\n\n---\n\n")
                else:
                    consolidated_f.write(f"## {task.upper()} Results\n\n")
                    consolidated_f.write(f"*No comparison data available for {task}*\n\n")
                    if task_idx < len(tasks) - 1:
                        consolidated_f.write("---\n\n")
            
            # Add OpenLLM Leaderboard style summary table
            consolidated_f.write("\n\n---\n\n")
            consolidated_f.write("## OpenLLM Leaderboard Style Summary\n\n")
            
            # Create the header
            header = "| Model |"
            separator = "|-------|"
            for task in tasks:
                header += f" {task.upper()} |"
                separator += "--------|"
            header += " Average |"
            separator += "---------|"
            
            consolidated_f.write(header + "\n")
            consolidated_f.write(separator + "\n")
            
            # Process each model
            for model_name in models:
                if model_name in leaderboard_data:
                    row = f"| {model_name} |"
                    task_scores = []
                    
                    for task in tasks:
                        if task in leaderboard_data[model_name]:
                            model_data = leaderboard_data[model_name][task]
                            accuracy = model_data.get('accuracy', 0)
                            recovery_pct = model_data.get('recovery_pct')
                            flips_pct = model_data.get('flips_pct')
                            
                            # Format the cell with accuracy and optional recovery/flips info
                            cell_content = f"{accuracy:.4f}"
                            if recovery_pct is not None and flips_pct is not None:
                                cell_content += f" ({recovery_pct:.2f}%, {flips_pct:.2f}% flips)"
                            
                            row += f" {cell_content} |"
                            task_scores.append(accuracy)
                        else:
                            row += " - |"
                    
                    # Calculate and add average
                    if task_scores:
                        avg_score = sum(task_scores) / len(task_scores)
                        row += f" {avg_score:.4f} |"
                    else:
                        row += " - |"
                    
                    consolidated_f.write(row + "\n")
            
            # Add explanation
            consolidated_f.write("\n**Table Legend:**\n")
            consolidated_f.write("- Accuracy scores are shown as decimal values\n")
            consolidated_f.write("- For quantized models: (Recovery%, Flips%) are shown in parentheses\n")
            consolidated_f.write("- Recovery%: Percentage of base model accuracy retained\n")
            consolidated_f.write("- Flips%: Percentage of predictions that changed from base model\n")
        
        print(f"\nConsolidated comparison saved to: {consolidated_path}")


if __name__ == "__main__":
    config = {
        "task_name": "openllm",
        "log_samples": True,
        "batch_size": 16,
        "num_fewshot": 0,
    }
    model_groups = [
        (
            "meta-llama/Llama-3.1-8B-Instruct",
            "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
            "DESUCLUB/Llama-3.1-8B-Instruct-bf16-quantized.w8a8",
        )
    ]
    task_results = {}
    group_results = {}
    group_dirs = {
        models[0]: f"model_group_{extract_base_model_name(models[0])}"
        for models in model_groups
    }
    vllm_models = {
        models[0]: is_vllm_quantized_model(model)
        for models in model_groups
        for model in models
    }
    PATH = ["zeroshot_server_eval"]
    for idx, group in enumerate(group_dirs):
        for model_name in os.listdir(os.path.join(PATH[idx], group_dirs[group])):
            md_file = os.path.join(
                PATH[idx],
                group_dirs[group],
                model_name,
                f"{config['task_name']}_results.md",
            )
            if os.path.exists(md_file):
                group_result = results_md_to_json(md_file, save=False)
                group_results[re.sub(r"__", "/", model_name)] = group_result
            else:
                print(f"WARNING: {md_file} does not exist")
                continue
        task_results[group] = group_results
    print(task_results)
    if config["task_name"] == "openllm":
        tasks_to_process = [
            "mmlu",
            "hellaswag",
            "winogrande",
            "truthfulqa_mc2",
            "arc_challenge",
            "gsm8k",
        ]
        for task in tasks_to_process:
            config["task_name"] = task
            _run_flip_analysis(
                config,
                model_groups,
                group_dirs,
                task_results,
                vllm_models,
                kl_divergence=False,
                PATH=PATH,
            )
        
        # Consolidate all task markdown files into a single comprehensive report
        print("\nCreating consolidated comparison report...")
        consolidate_task_markdowns(model_groups, group_dirs, PATH, tasks_to_process)
    else:
        _run_flip_analysis(
            config,
            model_groups,
            group_dirs,
            task_results,
            vllm_models,
            kl_divergence=False,
            PATH=PATH,
        )
