import json
import os
import re
import glob
from typing import Dict, List, Tuple, Set, Optional, Union


class FlipEval:
    """
    Compute flips (correct→wrong and wrong→correct) between two lm-eval-harness JSONL outputs.
    Can analyze individual files or entire directories of task results.
    """

    def __init__(self, base_path: str = None, new_path: str = None, task_name: str = None):
        """
        Initialize the FlipEval class with either individual files or directories.
        
        Args:
            base_path: Path to base model JSONL file or directory containing JSONL files
            new_path: Path to new model JSONL file or directory containing JSONL files
            task_name: Optional task name override
        """
        self.base_path = base_path
        self.new_path = new_path
        self.task_name = task_name
        
        self.is_directory_mode = False
        self.file_pairs = []
        
        # Only initialize if paths are provided
        if base_path and new_path:
            if os.path.isdir(base_path) and os.path.isdir(new_path):
                self.is_directory_mode = True
                self._find_matching_files()
            else:
                self._initialize_file_comparison(base_path, new_path)

    def _find_matching_files(self):
        """
        Find matching JSONL files in base and new directories.
        Pairs files that belong to the same task based on naming patterns.
        """
        base_files = glob.glob(os.path.join(self.base_path, "samples_*.jsonl"))
        
        for base_file in base_files:
            base_filename = os.path.basename(base_file)
            # Extract task pattern from filename (e.g., "samples_mmlu_anatomy.jsonl" -> "mmlu_anatomy")
            task_pattern = re.search(r"samples_([^_]+(?:_[^_]+)*)_\d{4}", base_filename)
            
            if task_pattern:
                task_id = task_pattern.group(1)
                # Look for matching file in new directory
                new_files = glob.glob(os.path.join(self.new_path, f"samples_{task_id}_*.jsonl"))
                
                if new_files:
                    # Use the most recent file if multiple matches
                    new_file = sorted(new_files)[-1]
                    self.file_pairs.append((base_file, new_file, task_id))
                    print(f"Found matching file pair for task {task_id}")
            else:
                print(f"Could not identify task pattern in filename: {base_filename}")
        
        print(f"Found {len(self.file_pairs)} matching file pairs")
    
        base_jsonl_files = [pair[0] for pair in self.file_pairs]
        new_jsonl_files = [pair[1] for pair in self.file_pairs]
        
        base_jsonl_files.sort()
        new_jsonl_files.sort()

    def _initialize_file_comparison(self, base_file_path: str, new_file_path: str):
        """
        Initialize for single file comparison.
        """
        # If task_name not provided, try to extract it from file path
        if self.task_name is None:
            self.task_name = self._extract_task_name(base_file_path)

        # Load raw JSON objects keyed by doc_id
        self.base = self._load_jsonl(base_file_path)
        self.new = self._load_jsonl(new_file_path)

        # Determine which binary signal to use ("acc" or "exact_match")
        self.signal_key = self._detect_signal_key(self.base)
        
        # Extract subtask information if available
        self.subtasks = self._extract_subtasks()

    def analyze_all_tasks(self) -> Dict:
        """
        Analyze all matching file pairs found in directories.
        Returns a dictionary with results for each task.
        """
        if not self.is_directory_mode:
            raise ValueError("This method requires directory mode. Initialize with directories, not files.")
        
        all_results = {}
        all_flips = {"total_examples": 0, "correct_to_wrong": 0, "wrong_to_correct": 0}
        
        for base_file, new_file, task_id in self.file_pairs:
            print(f"Analyzing task: {task_id}")
            
            # Create a new FlipEval instance for this file pair
            evaluator = FlipEval(base_file, new_file, task_name=task_id)
            stats = evaluator.count_flips()
            
            all_results[task_id] = stats
            
            # Aggregate totals
            all_flips["total_examples"] += stats["total_examples"]
            all_flips["correct_to_wrong"] += stats["correct_to_wrong"]
            all_flips["wrong_to_correct"] += stats["wrong_to_correct"]
        
        # Calculate overall metrics
        total_flips = all_flips["correct_to_wrong"] + all_flips["wrong_to_correct"]
        percent_flips = (total_flips / all_flips["total_examples"] * 100) if all_flips["total_examples"] else 0.0
        
        all_flips["total_flips"] = total_flips
        all_flips["percent_flips"] = percent_flips
        
        return {
            "tasks": all_results,
            "summary": all_flips
        }
        
    def print_all_task_summary(self):
        """
        Print a summary table of all tasks and their flip statistics.
        """
        if not self.is_directory_mode:
            raise ValueError("This method requires directory mode. Initialize with directories, not files.")
        
        results = self.analyze_all_tasks()
        summary = results["summary"]
        
        print("\n" + "=" * 80)
        print(f"OVERALL SUMMARY: {len(results['tasks'])} Tasks")
        print(f"Total examples: {summary['total_examples']}")
        print(f"Total flips: {summary['total_flips']} ({summary['percent_flips']:.2f}%)")
        print(f"  Correct → Wrong: {summary['correct_to_wrong']}")
        print(f"  Wrong → Correct: {summary['wrong_to_correct']}")
        print("=" * 80)
        
        print(f"\n{'Task':<30} {'Examples':<10} {'Flips':<10} {'C→W':<10} {'W→C':<10} {'%':<10}")
        print("-" * 80)
        
        # Sort tasks by flip percentage
        sorted_tasks = sorted(results["tasks"].items(), 
                             key=lambda x: x[1]["percent_flips"],
                             reverse=True)
        
        for task_id, stats in sorted_tasks:
            print(f"{task_id:<30} {stats['total_examples']:<10} {stats['total_flips']:<10} "
                 f"{stats['correct_to_wrong']:<10} {stats['wrong_to_correct']:<10} "
                 f"{stats['percent_flips']:.2f}%")

    def _extract_task_name(self, file_path: str) -> str:
        """
        Extract task name from the file path.
        """
        # Try to extract from "samples_TASKNAME_..." pattern
        filename = os.path.basename(file_path)
        match = re.search(r"samples_([^_]+)", filename)
        if match:
            return match.group(1)
        return "unknown"
    
    def _extract_subtasks(self) -> set:
        """
        Extract subtask names from JSONL objects if available.
        """
        subtasks = set()
        
        for doc_id, obj in self.base.items():
            # Look for subtask in doc fields
            if "doc" in obj:
                # Check various common subtask field names
                for field in ["subtask", "sub_task", "task", "category"]:
                    if field in obj["doc"]:
                        subtasks.add(obj["doc"][field])
                
                # If the subtask might be in metadata or part of another structure
                if "metadata" in obj["doc"] and isinstance(obj["doc"]["metadata"], dict):
                    for field in ["subtask", "sub_task", "task", "category"]:
                        if field in obj["doc"]["metadata"]:
                            subtasks.add(obj["doc"]["metadata"][field])
            
        # If we couldn't find subtasks in the doc objects, try to extract from filename
        if not subtasks:
            filename = os.path.basename(self.base_path)
            match = re.search(r"samples_([^_]+)_([^_]+)", filename)
            if match and match.group(2):
                subtasks.add(match.group(2))
        
        return subtasks

    def _load_jsonl(self, path: str) -> dict[int, dict]:
        """
        Load a JSONL file into a dict mapping doc_id → full JSON object.
        """
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = obj["doc_id"]
                data[doc_id] = obj
        return data

    def _detect_signal_key(self, sample_map: dict[int, dict]) -> str:
        """
        Inspect the first example's metrics list to pick a binary metric.
        For generative tasks, fall back to a continuous metric or skip if unavailable.
        """
        first = next(iter(sample_map.values()))
        metrics = first.get("metrics", [])
        
        # First, try to find common binary accuracy metrics
        if "acc" in metrics:
            return "acc"
        if "exact_match" in metrics:
            return "exact_match"
            
        # For TruthfulQA generative mode and similar tasks, try *_acc metrics 
        # which are the closest to binary metrics
        for metric in metrics:
            if metric.endswith("_acc"):
                print(f"Warning: No binary metric found. Using {metric} as fallback for flip analysis.")
                return metric
        
        # If we can't find a binary or *_acc metric, print a warning and return None
        # This will cause the task to be skipped in count_flips
        print(f"Warning: No appropriate metric found for flip analysis in metrics: {metrics}")
        print("This task will be skipped for flip analysis.")
        return None
    
    def _get_subtask_for_doc(self, doc_obj: dict) -> str:
        """
        Extract the subtask name for a given document.
        """
        # Check for subtask in doc fields
        if "doc" in doc_obj:
            # Check various common subtask field names
            for field in ["subtask", "sub_task", "task", "category"]:
                if field in doc_obj["doc"]:
                    return doc_obj["doc"][field]
            
            # If the subtask might be in metadata
            if "metadata" in doc_obj["doc"] and isinstance(doc_obj["doc"]["metadata"], dict):
                for field in ["subtask", "sub_task", "task", "category"]:
                    if field in doc_obj["doc"]["metadata"]:
                        return doc_obj["doc"]["metadata"][field]
        
        # If no subtask found, try to extract from task path
        filename = os.path.basename(self.base_path)
        match = re.search(r"samples_([^_]+)_([^_]+)", filename)
        if match and match.group(2):
            return match.group(2)
        
        # Default to unknown or task name
        return "unknown"

    def count_flips(self, verbose: bool = False) -> dict[str, int | float | list | dict]:
        """
        Compare base vs. new, assert same docs, and count flips.
        Returns a dict with:
          - total_examples
          - total_flips
          - percent_flips
          - correct_to_wrong
          - wrong_to_correct
          - per_subtask: flips broken down by subtask
          - correct_to_wrong_samples (if verbose=True)
          - wrong_to_correct_samples (if verbose=True)
        """
        # Make sure data is loaded
        if not hasattr(self, 'base') or not hasattr(self, 'new'):
            self._initialize_file_comparison(self.base_path, self.new_path)
        
        # For generative tasks or tasks without appropriate binary metrics,
        # we can't calculate flips in the traditional sense
        if self.signal_key is None:
            print(f"Skipping flip analysis for task {self.task_name} - no appropriate binary metric found")
            # Return empty results structure
            result = {
                "task_name": self.task_name,
                "total_examples": len(self.base),
                "total_flips": 0,
                "percent_flips": 0.0,
                "correct_to_wrong": 0,
                "wrong_to_correct": 0,
                "per_subtask": {"unknown": {
                    "total": len(self.base),
                    "correct_to_wrong": 0,
                    "wrong_to_correct": 0,
                    "total_flips": 0,
                    "percent_flips": 0.0
                }}
            }
            if verbose:
                result["correct_to_wrong_samples"] = []
                result["wrong_to_correct_samples"] = []
            return result
        
        correct_to_wrong = 0
        wrong_to_correct = 0
        matched = 0
        
        # For storing the actual flipped samples
        correct_to_wrong_samples = []
        wrong_to_correct_samples = []
        
        # Track flips per subtask
        per_subtask = {}
        
        for doc_id, base_obj in self.base.items():
            # Ensure the same doc_id exists in new run
            assert doc_id in self.new, f"doc_id {doc_id} missing in new file"
            new_obj = self.new[doc_id]

            # Instance-level assert: ensure same document by comparing doc_hash
            base_hash = base_obj.get("doc_hash")
            new_hash = new_obj.get("doc_hash")
            assert base_hash == new_hash, (
                f"doc_hash mismatch for doc_id {doc_id}: "
                f"{base_hash} (base) != {new_hash} (new)"
            )
            
            # Get subtask for this document
            subtask = self._get_subtask_for_doc(base_obj)
            
            # Initialize subtask counters if not exists
            if subtask not in per_subtask:
                per_subtask[subtask] = {
                    "total": 0,
                    "correct_to_wrong": 0,
                    "wrong_to_correct": 0,
                    "total_flips": 0,
                    "percent_flips": 0.0
                }
            
            # Update subtask counter
            per_subtask[subtask]["total"] += 1

            # Extract binary signals
            b = int(base_obj[self.signal_key])
            n = int(new_obj[self.signal_key])

            matched += 1
            if b != n:
                gold_idx = int(base_obj["doc"]["gold"]) if "gold" in base_obj["doc"] else None
                
                # Get predictions and scores
                base_pred = self._get_prediction(base_obj)
                new_pred = self._get_prediction(new_obj)
                
                if b == 1 and n == 0:
                    correct_to_wrong += 1
                    per_subtask[subtask]["correct_to_wrong"] += 1
                    
                    if verbose:
                        sample_data = {
                            "doc_id": doc_id,
                            "subtask": subtask,
                            "query": base_obj["doc"]["query"] if "query" in base_obj["doc"] else "",
                            "choices": base_obj["doc"]["choices"] if "choices" in base_obj["doc"] else [],
                            "gold": base_obj["doc"]["gold"] if "gold" in base_obj["doc"] else "",
                            "gold_text": base_obj["doc"]["choices"][gold_idx] if gold_idx is not None and "choices" in base_obj["doc"] else "",
                            "base_prediction": base_pred,
                            "new_prediction": new_pred,
                            "base_scores": self._get_scores(base_obj),
                            "new_scores": self._get_scores(new_obj)
                        }
                        correct_to_wrong_samples.append(sample_data)
                else:
                    wrong_to_correct += 1
                    per_subtask[subtask]["wrong_to_correct"] += 1
                    
                    if verbose:
                        sample_data = {
                            "doc_id": doc_id,
                            "subtask": subtask,
                            "query": base_obj["doc"]["query"] if "query" in base_obj["doc"] else "",
                            "choices": base_obj["doc"]["choices"] if "choices" in base_obj["doc"] else [],
                            "gold": base_obj["doc"]["gold"] if "gold" in base_obj["doc"] else "",
                            "gold_text": base_obj["doc"]["choices"][gold_idx] if gold_idx is not None and "choices" in base_obj["doc"] else "",
                            "base_prediction": base_pred,
                            "new_prediction": new_pred,
                            "base_scores": self._get_scores(base_obj),
                            "new_scores": self._get_scores(new_obj)
                        }
                        wrong_to_correct_samples.append(sample_data)

        total_flips = correct_to_wrong + wrong_to_correct
        percent_flips = (total_flips / matched * 100) if matched else 0.0

        # Calculate statistics per subtask
        for subtask, stats in per_subtask.items():
            stats["total_flips"] = stats["correct_to_wrong"] + stats["wrong_to_correct"]
            stats["percent_flips"] = (stats["total_flips"] / stats["total"] * 100) if stats["total"] else 0.0

        result = {
            "task_name": self.task_name,
            "total_examples": matched,
            "total_flips": total_flips,
            "percent_flips": percent_flips,
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
            "per_subtask": per_subtask
        }
        
        if verbose:
            result["correct_to_wrong_samples"] = correct_to_wrong_samples
            result["wrong_to_correct_samples"] = wrong_to_correct_samples
            
        return result
    
    def _get_prediction(self, obj):
        """Extract the model's prediction from the object"""
        # First check if there's a direct prediction field
        if "pred" in obj:
            return obj["pred"]
        
        # Otherwise determine from filtered_resps if available
        if "filtered_resps" in obj and isinstance(obj["filtered_resps"], list):
            # Find index of highest scoring option
            try:
                scores = [float(resp[0]) for resp in obj["filtered_resps"]]
                return scores.index(max(scores))
            except (ValueError, IndexError):
                pass
        
        return None
    
    def _get_scores(self, obj):
        """Extract the model's scores from the object"""
        if "filtered_resps" in obj:
            return obj["filtered_resps"]
        if "resps" in obj:
            return obj["resps"]
        return None

    def print_subtask_summary(self):
        """
        Print a summary of flips per subtask.
        """
        stats = self.count_flips()
        
        print(f"Task: {stats['task_name']}")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Overall flips: {stats['total_flips']} ({stats['percent_flips']:.2f}%)")
        print(f"  Correct → Wrong: {stats['correct_to_wrong']}")
        print(f"  Wrong → Correct: {stats['wrong_to_correct']}")
        
        print("\nPer Subtask Analysis:")
        print("=" * 60)
        print(f"{'Subtask':<30} {'Total':<7} {'Flips':<7} {'C→W':<7} {'W→C':<7} {'%':<7}")
        print("-" * 60)
        
        for subtask, data in stats['per_subtask'].items():
            print(f"{subtask:<30} {data['total']:<7} {data['total_flips']:<7} "
                  f"{data['correct_to_wrong']:<7} {data['wrong_to_correct']:<7} "
                  f"{data['percent_flips']:.2f}%")

    def analyze_all_task_files(self, base_files: List[str], new_files: List[str], verbose: bool = False) -> Dict:
        """
        Analyze multiple file pairs for the same task (e.g., MMLU subtasks).
        Returns a combined analysis across all files.
        
        Args:
            base_files: List of base model JSONL files
            new_files: List of new model JSONL files
            verbose: Whether to include detailed samples in the output
            
        Returns:
            Combined statistics across all files
        """
        all_results = {
            "task_name": self.task_name,
            "total_examples": 0,
            "total_flips": 0,
            "correct_to_wrong": 0,
            "wrong_to_correct": 0,
            "per_subtask": {},
            "percent_flips": 0.0
        }
        
        if verbose:
            all_results["correct_to_wrong_samples"] = []
            all_results["wrong_to_correct_samples"] = []
        
        # Group files by subtask pattern if possible
        base_files_by_pattern = self._group_files_by_pattern(base_files)
        new_files_by_pattern = self._group_files_by_pattern(new_files)
        
        # Find matching patterns between base and new files
        common_patterns = set(base_files_by_pattern.keys()) & set(new_files_by_pattern.keys())
        
        print(f"Found {len(common_patterns)} matching subtask patterns")
        
        # For each matching pattern, compare the files
        for pattern in common_patterns:
            base_file = base_files_by_pattern[pattern]
            new_file = new_files_by_pattern[pattern]
            
            subtask_name = self._extract_subtask_from_filename(base_file)
            print(f"Analyzing subtask: {subtask_name}")
            print(f"  Base: {os.path.basename(base_file)}")
            print(f"  New: {os.path.basename(new_file)}")
            
            # Create a temporary FlipEval just for this file pair
            temp_eval = FlipEval(base_file, new_file, task_name=subtask_name)
            stats = temp_eval.count_flips(verbose=verbose)
            
            # Add subtask results to the overall results
            all_results["total_examples"] += stats["total_examples"]
            all_results["correct_to_wrong"] += stats["correct_to_wrong"]
            all_results["wrong_to_correct"] += stats["wrong_to_correct"]
            
            # Store the per-subtask data
            all_results["per_subtask"][subtask_name] = stats["per_subtask"].get("unknown", {})
            all_results["per_subtask"][subtask_name]["total"] = stats["total_examples"]
            all_results["per_subtask"][subtask_name]["correct_to_wrong"] = stats["correct_to_wrong"]
            all_results["per_subtask"][subtask_name]["wrong_to_correct"] = stats["wrong_to_correct"]
            all_results["per_subtask"][subtask_name]["total_flips"] = stats["total_flips"]
            all_results["per_subtask"][subtask_name]["percent_flips"] = stats["percent_flips"]
            
            # Collect detailed samples if requested
            if verbose:
                if "correct_to_wrong_samples" in stats:
                    all_results["correct_to_wrong_samples"].extend(stats["correct_to_wrong_samples"])
                if "wrong_to_correct_samples" in stats:
                    all_results["wrong_to_correct_samples"].extend(stats["wrong_to_correct_samples"])
        
        # Calculate total flips and percentage
        all_results["total_flips"] = all_results["correct_to_wrong"] + all_results["wrong_to_correct"]
        if all_results["total_examples"] > 0:
            all_results["percent_flips"] = (all_results["total_flips"] / all_results["total_examples"]) * 100
        
        return all_results
    
    def _group_files_by_pattern(self, files: List[str]) -> Dict[str, str]:
        """
        Group files by their subtask pattern to match corresponding files.
        Returns a dict mapping pattern -> file path
        """
        file_map = {}
        
        for file_path in files:
            filename = os.path.basename(file_path)
            
            # Try different pattern extraction strategies
            
            # For MMLU format: samples_mmlu_subtask_with_underscores_timestamp.jsonl
            # We need to handle cases like "high_school_computer_science"
            mmlu_pattern = re.search(r'samples_mmlu_(.+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl$', filename)
            if mmlu_pattern:
                subtask_with_timestamp = mmlu_pattern.group(1)
                # Remove the timestamp part (last section after underscore that matches timestamp pattern)
                subtask = re.sub(r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+$', '', subtask_with_timestamp)
                pattern_key = f"mmlu_{subtask}"
                file_map[pattern_key] = file_path
                continue
            
            # For format: samples_taskname_subtaskname_timestamp.jsonl (simple cases)
            pattern1 = re.search(r'samples_([^_]+)_([^_]+)_\d', filename)
            if pattern1:
                task, subtask = pattern1.groups()
                # Check if this looks like it might have more underscores (for non-MMLU tasks)
                remaining_filename = filename[pattern1.end()-1:]  # Start from the digit
                if '_' in subtask and not remaining_filename.startswith('_2'):  # Likely has more parts
                    # Try to extract the full subtask name up to the timestamp
                    full_match = re.search(r'samples_([^_]+)_(.+)_(\d{4}-\d{2}-\d{2}T[\d\-\.]+)\.jsonl$', filename)
                    if full_match:
                        task, full_subtask, timestamp = full_match.groups()
                        pattern_key = f"{task}_{full_subtask}"
                        file_map[pattern_key] = file_path
                        continue
                
                pattern_key = f"{task}_{subtask}"
                file_map[pattern_key] = file_path
                continue
                
            # For format: taskname_subtaskname_timestamp.jsonl
            pattern2 = re.search(r'([^_]+)_([^_]+)_\d', filename)
            if pattern2:
                task, subtask = pattern2.groups()
                pattern_key = f"{task}_{subtask}"
                file_map[pattern_key] = file_path
                continue
            
            # Fallback: just use the filename without timestamp as the key
            pattern_key = re.sub(r'_\d{4}-\d{2}-\d{2}T[\d\-\.]+\.jsonl$', '', filename)
            if pattern_key.startswith('samples_'):
                pattern_key = pattern_key[8:]  # Remove 'samples_' prefix
            file_map[pattern_key] = file_path
            
        return file_map
    
    def _extract_subtask_from_filename(self, file_path: str) -> str:
        """Extract subtask name from filename pattern"""
        filename = os.path.basename(file_path)
        
        # Try different patterns
        # For MMLU: samples_mmlu_high_school_computer_science_timestamp.jsonl -> "high_school_computer_science"
        mmlu_pattern = re.search(r'samples_mmlu_(.+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl$', filename)
        if mmlu_pattern:
            subtask_with_timestamp = mmlu_pattern.group(1)
            # Remove the timestamp part 
            subtask = re.sub(r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+$', '', subtask_with_timestamp)
            return subtask
        
        # samples_taskname_subtaskname_timestamp.jsonl -> "subtaskname" (for simple cases)
        pattern1 = re.search(r'samples_[^_]+_([^_]+)_\d', filename)
        if pattern1:
            return pattern1.group(1)
            
        # mmlu_anatomy_20250522_123456.jsonl -> "anatomy"
        pattern2 = re.search(r'[^_]+_([^_]+)_\d', filename)
        if pattern2:
            return pattern2.group(1)
            
        # If we can't determine a specific subtask, use a default
        if self.task_name:
            return self.task_name
            
        return "unknown"


if __name__ == "__main__":
    # Example 1: Single file comparison
    BASE_FILE_PATH = "/home/spooky/Documents/smol_projects/inference/lm-evaluation-harness/output/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/samples_mmlu_world_religions_2025-05-21T13-16-46.241074.jsonl"
    NEW_FILE_PATH = "/home/spooky/Documents/smol_projects/inference/lm-evaluation-harness/output/unsloth__DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit/samples_mmlu_world_religions_2025-05-21T13-26-19.911843.jsonl"
    
    print("Single file comparison example:")
    fe = FlipEval(BASE_FILE_PATH, NEW_FILE_PATH)
    fe.print_subtask_summary()
    
    # Example 2: Directory comparison
    BASE_DIR = "/home/spooky/Documents/smol_projects/inference/lm-evaluation-harness/output/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/"
    NEW_DIR = "/home/spooky/Documents/smol_projects/inference/lm-evaluation-harness/output/unsloth__DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit/"
    
    print("\nDirectory comparison example:")
    dir_fe = FlipEval(BASE_DIR, NEW_DIR)
    dir_fe.print_all_task_summary()
