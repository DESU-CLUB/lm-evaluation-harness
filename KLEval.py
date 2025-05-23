import torch
import os
from typing import Dict, List, Any, Optional
import glob
import json

class KLEval:
    def __init__(self, base_model_dir: str, new_model_dir: str):
        """
        Initialize KLEval with paths to the model directories containing manifest.pt or task-specific tensor files
        
        Args:
            base_model_dir: Directory for base model tensor data
            new_model_dir: Directory for new (quantized) model tensor data
        """
        self.base_model_dir = base_model_dir
        self.new_model_dir = new_model_dir
        
        # Load manifests if they exist or find task files directly
        self.base_manifest = self._load_manifest(base_model_dir)
        self.new_manifest = self._load_manifest(new_model_dir)

    def _load_manifest(self, model_dir: str) -> Dict[str, Any]:
        """Load manifest.pt file or gather task files if no manifest exists"""
        manifest_path = os.path.join(model_dir, "manifest.pt")
        
        if os.path.exists(manifest_path):
            return torch.load(manifest_path)
        else:
            # Try to find task files directly
            task_files = glob.glob(os.path.join(model_dir, "*.pt"))
            # Exclude any manifest file that might be in the list
            task_files = [f for f in task_files if os.path.basename(f) != "manifest.pt"]
            
            if not task_files:
                # Look for files in a *_logits subdirectory
                logit_dirs = [d for d in os.listdir(model_dir) if d.endswith("_logits") and os.path.isdir(os.path.join(model_dir, d))]
                for logit_dir in logit_dirs:
                    task_files.extend(glob.glob(os.path.join(model_dir, logit_dir, "*.pt")))
                
                # Look for *_tensors subdirectories (new format)
                tensor_dirs = [d for d in os.listdir(model_dir) if d.endswith("_tensors") and os.path.isdir(os.path.join(model_dir, d))]
                task_files_from_tensors = []
                
                for tensor_dir in tensor_dirs:
                    # Check for manifest.json in the tensor directory
                    tensor_dir_path = os.path.join(model_dir, tensor_dir)
                    manifest_json_path = os.path.join(tensor_dir_path, "manifest.json")
                    
                    if os.path.exists(manifest_json_path):
                        # Use the manifest to get task info
                        with open(manifest_json_path, 'r') as f:
                            tensor_manifest = json.load(f)
                        
                        task_name = tensor_manifest.get('task_name', os.path.basename(tensor_dir).replace("_tensors", ""))
                        task_files_from_tensors.append({
                            "task_name": task_name,
                            "is_tensor_dir": True,
                            "dir_path": tensor_dir_path,
                            "tensor_count": tensor_manifest.get('tensor_count', 0)
                        })
                    else:
                        # No manifest, try to infer from directory structure
                        task_name = os.path.basename(tensor_dir).replace("_tensors", "")
                        # Count tensor files
                        tensor_files = glob.glob(os.path.join(tensor_dir_path, "tensor_*.pt"))
                        task_files_from_tensors.append({
                            "task_name": task_name,
                            "is_tensor_dir": True,
                            "dir_path": tensor_dir_path,
                            "tensor_count": len(tensor_files)
                        })
            
            if not task_files and not task_files_from_tensors:
                raise FileNotFoundError(f"No manifest.pt, task tensor files, or tensor directories found in {model_dir}")
                
            # Create a synthetic manifest
            if task_files:
                # Old format with individual task files
                return {
                    "model": os.path.basename(model_dir),
                    "task_files": [
                        {"task_name": os.path.basename(f).replace(".pt", ""), "file": f}
                        for f in task_files
                    ]
                }
            else:
                # New format with tensor directories
                return {
                    "model": os.path.basename(model_dir),
                    "task_files": task_files_from_tensors
                }
    
    def _get_matching_task_files(self, task_filter: str = None) -> List[Dict[str, Any]]:
        """
        Match task files between base and new model
        
        Args:
            task_filter: Optional filter to only process specific task(s)
        """
        # Filter out manifest.pt from task files and extract task names
        base_tasks = {}
        for task_file in self.base_manifest.get("task_files", []):
            task_name = task_file["task_name"]
            if task_name.lower() != "manifest":  # Explicitly filter out manifest.pt
                base_tasks[task_name] = task_file
        
        new_tasks = {}
        for task_file in self.new_manifest.get("task_files", []):
            task_name = task_file["task_name"]
            if task_name.lower() != "manifest":  # Explicitly filter out manifest.pt
                new_tasks[task_name] = task_file
        
        # Apply task filter if specified
        if task_filter:
            base_tasks = {name: info for name, info in base_tasks.items() if task_filter in name}
            new_tasks = {name: info for name, info in new_tasks.items() if task_filter in name}
            print(f"Filtered to tasks matching '{task_filter}'")
        
        print(f"Base model tasks after filtering: {list(base_tasks.keys())}")
        print(f"New model tasks after filtering: {list(new_tasks.keys())}")
        
        # Find common tasks
        common_tasks = set(base_tasks.keys()) & set(new_tasks.keys())
        
        if not common_tasks:
            raise ValueError(f"No common tasks found between base and new models{' matching filter ' + task_filter if task_filter else ''}")
            
        print(f"Found {len(common_tasks)} matching tasks: {sorted(list(common_tasks))}")
        
        return [
            {
                "task_name": task_name,
                "base_info": base_tasks[task_name],
                "new_info": new_tasks[task_name]
            }
            for task_name in common_tasks
        ]
    
    def _load_tensors_from_directory(self, tensor_dir: str, start_idx: int = 0, end_idx: int = None) -> List[torch.Tensor]:
        """
        Load tensors from a directory containing individual tensor files
        
        Args:
            tensor_dir: Directory containing tensor_*.pt files
            start_idx: Starting index for tensors to load
            end_idx: Ending index for tensors to load (None means load all)
            
        Returns:
            List of loaded tensors
        """
        if end_idx is None:
            # Try to get tensor count from manifest
            manifest_path = os.path.join(tensor_dir, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                end_idx = manifest.get("tensor_count", 0)
            
            # If still None, count files
            if end_idx is None or end_idx == 0:
                tensor_files = glob.glob(os.path.join(tensor_dir, "tensor_*.pt"))
                end_idx = len(tensor_files)
        
        tensors = []
        for i in range(start_idx, end_idx):
            tensor_path = os.path.join(tensor_dir, f"tensor_{i:06d}.pt")
            if os.path.exists(tensor_path):
                tensors.append(torch.load(tensor_path))
            else:
                print(f"Warning: Missing tensor file {tensor_path}")
        
        return tensors

    @torch.no_grad()
    def calculate_per_task_kl(self, task_filter: str = None, return_details: bool = False) -> Dict[str, Any]:
        """
        Calculate KL divergence per task, handling MCQ tasks differently
        
        Args:
            task_filter: Optional filter string to only process specific tasks
            return_details: Whether to include detailed per-sample KL values
        
        Returns:
            Dictionary with KL divergence results
        """
        results = {}
        matched_tasks = self._get_matching_task_files(task_filter=task_filter)
        
        # Process each task
        for task_info in matched_tasks:
            task_name = task_info["task_name"]
            print(f"Processing task: {task_name}")
            
            try:
                # Handle both old and new tensor formats
                base_info = task_info["base_info"]
                new_info = task_info["new_info"]
                
                # Check if this is a tensor directory format
                if "is_tensor_dir" in base_info and base_info["is_tensor_dir"]:
                    # New format with individual tensor files
                    print(f"Loading tensors from directories")
                    base_tensors = self._load_tensors_from_directory(base_info["dir_path"])
                    new_tensors = self._load_tensors_from_directory(new_info["dir_path"])
                    
                    # Create synthetic task data
                    base_task_data = {
                        "task_name": task_name,
                        "task_type": "multiple_choice" if "hellaswag" in task_name.lower() else "generative",
                        "tensors": base_tensors
                    }
                    
                    new_task_data = {
                        "task_name": task_name,
                        "task_type": base_task_data["task_type"],
                        "tensors": new_tensors
                    }
                    
                    # For MCQ tasks like hellaswag, try to determine options_per_question
                    if base_task_data["task_type"] == "multiple_choice":
                        # For hellaswag, each question typically has 4 options
                        base_task_data["options_per_question"] = [4] * (len(base_tensors) // 4)
                        if len(base_tensors) % 4 != 0:
                            print(f"Warning: {task_name} has {len(base_tensors)} tensors, which is not a multiple of 4. Handling remainder.")
                            # Add the remainder
                            base_task_data["options_per_question"].append(len(base_tensors) % 4)
                else:
                    # Load task tensors from old format
                    base_file = base_info["file"]
                    new_file = new_info["file"]
                    base_task_data = torch.load(base_file)
                    new_task_data = torch.load(new_file)
                
                # Check for task type consistency
                if "task_type" not in base_task_data:
                    print(f"Warning: No task_type in {task_name}, assuming generative")
                    task_type = "generative"
                else:
                    task_type = base_task_data["task_type"]
                
                # Debug info
                print(f"Base task data keys: {base_task_data.keys()}")
                
                # Check if tensors exist in the data
                if "tensors" not in base_task_data or "tensors" not in new_task_data:
                    print(f"Error: 'tensors' key not found in task data for {task_name}")
                    continue
                
                # Extract tensors
                base_task_tensors = base_task_data["tensors"]
                new_task_tensors = new_task_data["tensors"]
                
                if not base_task_tensors or not new_task_tensors:
                    print(f"Error: Empty tensors for task {task_name}")
                    continue
                
                print(f"Found {len(base_task_tensors)} base tensors and {len(new_task_tensors)} new tensors")
                
                if len(base_task_tensors) != len(new_task_tensors):
                    print(f"Warning: Different number of samples for task {task_name}: {len(base_task_tensors)} vs {len(new_task_tensors)}")
                    continue
                
                # For MCQ tasks, we need to know the number of options per question
                if task_type == "multiple_choice":
                    # This will track KL per question
                    question_kls = []
                    options_per_question = base_task_data.get("options_per_question", [])
                    
                    if not options_per_question:
                        print(f"Warning: No options_per_question found for MCQ task {task_name}")
                        # Try to infer from tensors
                        options_per_question = [len(base_task_tensors)]
                    
                    option_index = 0
                    for question_idx, num_options in enumerate(options_per_question):
                        # Process all options for this question
                        option_kls = []
                        for i in range(num_options):
                            if option_index >= len(base_task_tensors):
                                break
                                
                            base_sample = base_task_tensors[option_index]
                            new_sample = new_task_tensors[option_index]
                            
                            # Calculate KL for this option
                            # Debug info for first tensor
                            if question_idx == 0 and i == 0:
                                print(f"Sample tensor type: {type(base_sample)}")
                                print(f"Sample tensor shape: {base_sample.shape}")
                            
                            if base_sample.shape != new_sample.shape:
                                print(f"Skipping option with mismatched shape in question {question_idx}: {base_sample.shape} vs {new_sample.shape}")
                                option_index += 1
                                continue
                            
                            # Convert to log probabilities and calculate per-token KL
                            base_log_probs = base_sample.log_softmax(dim=-1)
                            new_log_probs = new_sample.log_softmax(dim=-1)
                            token_kls = torch.nn.functional.kl_div(
                                new_log_probs,
                                base_log_probs.exp(),
                                reduction='none'
                            ).sum(dim=-1)  # Sum across vocabulary
                            
                            # Mean KL for this option
                            option_kl = token_kls.mean().item()
                            option_kls.append(option_kl)
                            option_index += 1
                        
                        # Calculate average KL for this question (across all options)
                        if option_kls:
                            question_kl = sum(option_kls) / len(option_kls)
                            question_kls.append(question_kl)
                    
                    # Task-level KL is mean of all question KLs
                    task_kl = sum(question_kls) / max(1, len(question_kls))
                    results[task_name] = {
                        "mean_kl": task_kl,
                        "type": "multiple_choice",
                        "num_questions": len(question_kls),
                    }
                    if return_details:
                        results[task_name]["per_question_kls"] = question_kls
                else:
                    # For generative tasks - simpler calculation
                    sample_kls = []
                    for idx, (base_sample, new_sample) in enumerate(zip(base_task_tensors, new_task_tensors)):
                        # Debug info for first tensor
                        if idx == 0:
                            print(f"Sample tensor shape: {base_sample.shape}")
                        
                        if base_sample.shape != new_sample.shape:
                            print(f"Skipping sample with mismatched shapes in task {task_name}")
                            continue
                        
                        # Convert to log probabilities
                        base_log_probs = base_sample.log_softmax(dim=-1)
                        new_log_probs = new_sample.log_softmax(dim=-1)
                        
                        # Sum KL per token first (across vocabulary)
                        token_kls = torch.nn.functional.kl_div(
                            new_log_probs,
                            base_log_probs.exp(),
                            reduction='none'
                        ).sum(dim=-1)
                        
                        # Mean across sequence
                        sample_kl = token_kls.mean().item()
                        sample_kls.append(sample_kl)
                    
                    # Task-level KL is mean of all sample KLs
                    task_kl = sum(sample_kls) / max(1, len(sample_kls))
                    results[task_name] = {
                        "mean_kl": task_kl,
                        "type": "generative",
                        "num_samples": len(sample_kls),
                    }
                    if return_details:
                        results[task_name]["per_sample_kls"] = sample_kls
                
                # Clean up to save memory
                del base_task_tensors
                del new_task_tensors
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing task {task_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate overall KL across all tasks
        if results:
            overall_kl = sum(task["mean_kl"] for task in results.values()) / len(results)
            results["overall"] = {"mean_kl": overall_kl}
        
        return results

    def save_kl_results(self, output_path: str = None, task_filter: str = None):
        """
        Calculate KL divergence and save results to files (.pt and .json)
        
        Args:
            output_path: Path to save results (without extension)
            task_filter: Optional filter to only process specific tasks
            
        Returns:
            Dictionary with KL divergence results
        """
        results = self.calculate_per_task_kl(task_filter=task_filter, return_details=True)
        
        if output_path is None:
            # Create default output path
            base_model_name = os.path.basename(self.base_model_dir)
            new_model_name = os.path.basename(self.new_model_dir)
            task_suffix = f"_{task_filter}" if task_filter else ""
            output_path = f"kl_results_{base_model_name}_vs_{new_model_name}{task_suffix}"
        
        # Remove any extension from output_path
        output_path = os.path.splitext(output_path)[0]
        
        # Save results in PyTorch format
        pt_path = f"{output_path}.pt"
        torch.save(results, pt_path)
        print(f"KL results saved to {pt_path}")
        
        # Also save as JSON for easier inspection
        import json
        
        # Create a JSON-serializable version of the results
        json_results = {}
        for task_name, task_data in results.items():
            json_results[task_name] = {
                key: (value if not isinstance(value, torch.Tensor) else value.item() 
                     if value.numel() == 1 else value.tolist())
                for key, value in task_data.items()
            }
        
        json_path = f"{output_path}.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"JSON results saved to {json_path}")
        
        # Print a summary
        print("\nKL Divergence Summary:")
        print("=" * 50)
        for task_name, task_results in results.items():
            if task_name != "overall":
                print(f"{task_name}: {task_results['mean_kl']:.6f}")
        print("-" * 50)
        print(f"Overall KL: {results['overall']['mean_kl']:.6f}")
        
        return results
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate KL divergence between two models' outputs")
    parser.add_argument("base_model_dir", help="Directory with base model tensor files")
    parser.add_argument("new_model_dir", help="Directory with new model tensor files")
    parser.add_argument("--output", "-o", help="Output path for KL results")
    parser.add_argument("--task", "-t", help="Task filter (e.g., 'mmlu' or 'hellaswag')")
    
    args = parser.parse_args()
    
    kl_eval = KLEval(args.base_model_dir, args.new_model_dir)
    kl_eval.save_kl_results(args.output, task_filter=args.task)
        
        