import os
import json
import glob
from typing import Any, Dict, List, Optional

import torch


class KLEval:
    """Utility class to compute KL-divergence between two sets of logits saved by the
    updated evaluator.  It understands the *new* manifest.json layout but keeps
    backward-compatibility with the older .pt manifest and single-file format.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, base_model_dir: str, new_model_dir: str):
        self.base_root = os.path.abspath(base_model_dir)
        self.new_root = os.path.abspath(new_model_dir)

        self.base_manifest = self._load_root_manifest(self.base_root)
        self.new_manifest = self._load_root_manifest(self.new_root)

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------
    def _load_root_manifest(self, model_dir: str) -> Dict[str, Any]:
        """Load the *root* manifest that sits directly inside a model output dir.

        1.  Preferred:  JSON manifest produced by the new evaluator
        """
        json_path = os.path.join(model_dir, "manifest.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Expected manifest.json not found in {model_dir}. Old formats are no longer supported.")

        with open(json_path, "r") as f:
            m = json.load(f)

        # Normalise any relative dir_path fields to absolute paths
        for tf in m.get("task_files", []):
            if tf.get("is_tensor_dir") and "dir_path" in tf and not os.path.isabs(tf["dir_path"]):
                tf["dir_path"] = os.path.abspath(tf["dir_path"])

        return m

    # ------------------------------------------------------------------
    # Tensor loading helpers
    # ------------------------------------------------------------------
    def _load_tensors_from_task(self, task_entry: Dict[str, Any]) -> List[torch.Tensor]:
        """Return a flat list of tensors for a single task, loading lazily from
        chunk files or individual tensor files according to the entry.
        """
        # Expected new format: a tensor directory with its own manifest
        if not task_entry.get("is_tensor_dir"):
            raise ValueError("Only tensor-directory tasks are supported in the current format")

        tdir = task_entry["dir_path"]
        t_manifest_path = os.path.join(tdir, "manifest.json")
        if not os.path.exists(t_manifest_path):
            raise FileNotFoundError(f"Per-task manifest.json missing in {tdir}")

        with open(t_manifest_path, "r") as f:
            t_manifest = json.load(f)

        chunk_files = [c["new_file"] for c in t_manifest["chunks"]]
        print(chunk_files)

        tensors: List[torch.Tensor] = []
        for cf in chunk_files:
            chunk = torch.load(cf, map_location="cpu")
            if isinstance(chunk, list):
                tensors.extend(chunk)
            else:
                tensors.append(chunk)

        return tensors

    # ------------------------------------------------------------------
    # KL computation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
        """Compute mean token-level KL-divergence between two logits tensors.
        Shapes must match (seq, vocab) or (1, seq, vocab).  Returns scalar.
        """
        if p_logits.shape != q_logits.shape:
            raise ValueError("Logit shape mismatch: " + str((p_logits.shape, q_logits.shape)))

        # Convert to log probabilities and calculate per-token KL
        base_log_probs = p_logits.log_softmax(dim=-1)
        new_log_probs = q_logits.log_softmax(dim=-1)
        token_kls = torch.nn.functional.kl_div(
            new_log_probs,
            base_log_probs.exp(),
            reduction='none'
        ).sum(dim=-1)  # Sum across vocabulary
        
        # Mean KL for this option
        option_kl = token_kls.mean().item()
        return option_kl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate_per_task_kl(self, task_filter: Optional[str] = None, return_details: bool = False) -> Dict[str, Any]:
        """Compute KL for all common tasks between the two models."""
        # Build dicts task_name -> entry
        base_tasks = {t["task_name"]: t for t in self.base_manifest.get("task_files", [])}
        new_tasks = {t["task_name"]: t for t in self.new_manifest.get("task_files", [])}

        if task_filter:
            base_tasks = {k: v for k, v in base_tasks.items() if task_filter in k}
            new_tasks = {k: v for k, v in new_tasks.items() if task_filter in k}
            print(f"[INFO] Task filter='{task_filter}' → {list(base_tasks.keys())}")

        common = sorted(set(base_tasks) & set(new_tasks))
        if not common:
            raise ValueError("No common tasks between models after filtering")

        results: Dict[str, Any] = {}

        for task_name in common:
            print(f"[INFO] Processing task: {task_name}")
            try:
                base_entry = base_tasks[task_name]
                new_entry = new_tasks[task_name]

                base_tensors = self._load_tensors_from_task(base_entry)
                new_tensors = self._load_tensors_from_task(new_entry)

                if len(base_tensors) != len(new_tensors):
                    print(f"[WARN] Sample count mismatch for {task_name}: {len(base_tensors)} vs {len(new_tensors)} – truncating to min")
                    n = min(len(base_tensors), len(new_tensors))
                    base_tensors = base_tensors[:n]
                    new_tensors = new_tensors[:n]

                if not base_tensors:
                    print(f"[WARN] No tensors for task {task_name}; skipping")
                    continue

                # Determine task type & mcq metadata (from per-task manifest if available)
                task_type = "generative"
                options_per_question: Optional[List[int]] = None

                per_task_manifest_path = os.path.join(base_entry.get("dir_path", ""), "manifest.json")
                if os.path.exists(per_task_manifest_path):
                    with open(per_task_manifest_path, "r") as f:
                        ptm = json.load(f)
                    task_type = ptm.get("task_type", task_type)
                    options_per_question = ptm.get("options_per_question")

                # Multiple-choice KL: average per question
                if task_type == "multiple_choice":
                    if not options_per_question:
                        # Fallback heuristic: assume 4 options
                        options_per_question = [4] * (len(base_tensors) // 4)

                    q_kls: List[float] = []
                    idx = 0
                    for num_opts in options_per_question:
                        opts_kl = []
                        for _ in range(num_opts):
                            if idx >= len(base_tensors):
                                break
                            kl_val = self._kl_divergence(base_tensors[idx], new_tensors[idx])
                            opts_kl.append(kl_val)
                            idx += 1
                        if opts_kl:
                            q_kls.append(sum(opts_kl) / len(opts_kl))

                    mean_kl = sum(q_kls) / len(q_kls)
                    results[task_name] = {
                        "mean_kl": mean_kl,
                        "type": "multiple_choice",
                        "num_questions": len(q_kls),
                    }
                    if return_details:
                        results[task_name]["per_question_kls"] = q_kls

                else:  # generative
                    sample_kls = [self._kl_divergence(b, n) for b, n in zip(base_tensors, new_tensors)]
                    # Task-level KL is mean of all sample KLs
                    task_kl = sum(sample_kls) / max(1, len(sample_kls))
                    results[task_name] = {
                        "mean_kl": task_kl,
                        "type": "generative",
                        "num_samples": len(sample_kls),
                    }
                    if return_details:
                        results[task_name]["per_sample_kls"] = sample_kls

            except Exception as e:
                print(f"Error processing task {task_name}: {e}")

            # Cleanup memory between tasks
            del base_tensors, new_tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Calculate overall KL across all tasks
        if results:
            overall_kl = sum(task["mean_kl"] for task in results.values()) / len(results)
            results["overall"] = {"mean_kl": overall_kl}

        return results

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------
    def save_kl_results(self, output_path: Optional[str] = None, task_filter: Optional[str] = None):
        res = self.calculate_per_task_kl(task_filter=task_filter, return_details=True)

        if output_path is None:
            b_name = os.path.basename(self.base_root)
            n_name = os.path.basename(self.new_root)
            suffix = f"_{task_filter}" if task_filter else ""
            output_path = f"kl_results_{b_name}_vs_{n_name}{suffix}"

        output_path = os.path.splitext(output_path)[0]

        torch.save(res, f"{output_path}.pt")
        with open(f"{output_path}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"[INFO] Results saved to {output_path}.pt/.json")

        # Pretty summary
        print("\nKL Divergence Summary")
        print("=" * 40)
        for t, d in res.items():
            if t == "overall":
                continue
            print(f"{t:20s} : {d['mean_kl']:.6f}")
        print("-" * 40)
        print(f"OVERALL            : {res['overall']['mean_kl']:.6f}")

        return res

    # ------------------------------------------------------------------
    # Real-time KL computation for streaming logits
    # ------------------------------------------------------------------
    @staticmethod
    def compute_streaming_kl(base_logits_queue, new_logits_queue, buffer_size=100, task_metadata=None):
        """
        Compute KL divergence from two queues of logits.
        Collects all logits from both queues, converts to tensors, and computes KL.
        
        Args:
            base_logits_queue: Queue containing base model logits
            new_logits_queue: Queue containing new model logits  
            buffer_size: Unused (kept for compatibility)
            task_metadata: Dict with task_name, task_type, and MCQ metadata
            
        Returns:
            Dict with KL divergence statistics
        """
        import queue
        
        # Extract metadata
        if task_metadata is None:
            task_metadata = {'task_name': 'unknown', 'task_type': 'generative'}
        
        task_name = task_metadata.get('task_name', 'unknown')
        task_type = task_metadata.get('task_type', 'generative') 
        default_options = task_metadata.get('default_options', 4)
        
        base_logits_list = []
        new_logits_list = []
        
        print(f"[INFO] Collecting logits from queues for {task_name} (type: {task_type})")
        
        # Collect all logits from base model queue
        print(f"[INFO] Collecting base model logits...")
        while True:
            try:
                logits = base_logits_queue.get(timeout=60)  # Longer timeout for model completion
                if logits is None:  # Sentinel value indicates end of stream
                    print(f"[INFO] Base model finished, collected {len(base_logits_list)} logits")
                    break
                base_logits_list.append(logits)
                if len(base_logits_list) % 100 == 0:
                    print(f"[INFO] Base model: collected {len(base_logits_list)} logits")
            except queue.Empty:
                print(f"[WARN] Timeout waiting for base model logits")
                break
        
        # Collect all logits from new model queue  
        print(f"[INFO] Collecting new model logits...")
        while True:
            try:
                logits = new_logits_queue.get(timeout=60)
                if logits is None:  # Sentinel value indicates end of stream
                    print(f"[INFO] New model finished, collected {len(new_logits_list)} logits")
                    break
                new_logits_list.append(logits)
                if len(new_logits_list) % 100 == 0:
                    print(f"[INFO] New model: collected {len(new_logits_list)} logits")
            except queue.Empty:
                print(f"[WARN] Timeout waiting for new model logits")
                break
        
        # Check if we have matching numbers of logits
        if len(base_logits_list) != len(new_logits_list):
            print(f"[WARN] Mismatched logits count: base={len(base_logits_list)}, new={len(new_logits_list)}")
            # Truncate to minimum length
            min_len = min(len(base_logits_list), len(new_logits_list))
            base_logits_list = base_logits_list[:min_len]
            new_logits_list = new_logits_list[:min_len]
            print(f"[INFO] Truncated to {min_len} logits for comparison")
        
        if not base_logits_list or not new_logits_list:
            print(f"[ERROR] No logits collected for {task_name}")
            return {
                "task_name": task_name,
                "mean_kl": 0.0,
                "num_samples": 0,
                "type": "streaming",
                "error": "No logits collected"
            }
        
        # Compute KL divergence based on task type
        if task_type == "multiple_choice":
            options_per_question = task_metadata.get('options_per_question')
            
            if options_per_question:
                # Handle variable options per question
                print(f"[INFO] Computing MCQ KL divergence with variable options per question")
                question_kls = []
                idx = 0
                
                for q, num_opts in enumerate(options_per_question):
                    if idx + num_opts > len(base_logits_list):
                        print(f"[WARN] Not enough logits for question {q} (need {num_opts}, have {len(base_logits_list) - idx})")
                        break
                        
                    # Compute KL for each option in this question
                    option_kls = []
                    for opt in range(num_opts):
                        if idx >= len(base_logits_list):
                            break
                        try:
                            kl = KLEval._kl_divergence(base_logits_list[idx], new_logits_list[idx])
                            option_kls.append(kl)
                        except Exception as e:
                            print(f"[WARN] KL computation error for question {q}, option {opt}: {e}")
                        idx += 1
                    
                    # Average KL across options for this question
                    if option_kls:
                        question_kl = sum(option_kls) / len(option_kls)
                        question_kls.append(question_kl)
                        
                        if (q + 1) % 100 == 0:
                            current_mean = sum(question_kls) / len(question_kls)
                            print(f"[INFO] Processed {q + 1}/{len(options_per_question)} questions, current mean KL: {current_mean:.6f}")
                
                # Calculate final statistics
                if question_kls:
                    mean_kl = sum(question_kls) / len(question_kls)
                    min_kl = min(question_kls)
                    max_kl = max(question_kls)
                    
                    result = {
                        "task_name": task_name,
                        "mean_kl": mean_kl,
                        "min_kl": min_kl,
                        "max_kl": max_kl,
                        "num_questions": len(question_kls),
                        "num_samples": idx,  # Total options processed
                        "type": "multiple_choice",
                        "variable_options": True,
                        "options_per_question": options_per_question[:len(question_kls)]
                    }
                    
                    print(f"[INFO] Final variable MCQ KL stats for {task_name}:")
                    print(f"       Mean: {mean_kl:.6f}")
                    print(f"       Questions: {len(question_kls)}")
                    print(f"       Total options: {idx}")
                    
                    return result
                else:
                    return {
                        "task_name": task_name,
                        "mean_kl": 0.0,
                        "num_questions": 0,
                        "type": "multiple_choice",
                        "error": "No valid variable question KL computations"
                    }
            
            else:
                # Handle fixed options per question (original logic)
                print(f"[INFO] Computing MCQ KL divergence for {len(base_logits_list)} logits with {default_options} options per question")
                
                # Group logits by questions (assume default_options per question)
                num_questions = len(base_logits_list) // default_options
                if len(base_logits_list) % default_options != 0:
                    print(f"[WARN] Logits count {len(base_logits_list)} not divisible by {default_options}, truncating")
                    num_questions = len(base_logits_list) // default_options
                    base_logits_list = base_logits_list[:num_questions * default_options]
                    new_logits_list = new_logits_list[:num_questions * default_options]
                
                question_kls = []
                
                for q in range(num_questions):
                    start_idx = q * default_options
                    end_idx = start_idx + default_options
                    
                    # Compute KL for each option in this question
                    option_kls = []
                    for i in range(start_idx, end_idx):
                        try:
                            kl = KLEval._kl_divergence(base_logits_list[i], new_logits_list[i])
                            option_kls.append(kl)
                        except Exception as e:
                            print(f"[WARN] KL computation error for question {q}, option {i-start_idx}: {e}")
                            continue
                    
                    # Average KL across options for this question
                    if option_kls:
                        question_kl = sum(option_kls) / len(option_kls)
                        question_kls.append(question_kl)
                        
                        if (q + 1) % 100 == 0:
                            current_mean = sum(question_kls) / len(question_kls)
                            print(f"[INFO] Processed {q + 1}/{num_questions} questions, current mean KL: {current_mean:.6f}")
                
                # Calculate final statistics across questions
                if question_kls:
                    mean_kl = sum(question_kls) / len(question_kls)
                    min_kl = min(question_kls) 
                    max_kl = max(question_kls)
                    
                    result = {
                        "task_name": task_name,
                        "mean_kl": mean_kl,
                        "min_kl": min_kl,
                        "max_kl": max_kl,
                        "num_questions": len(question_kls),
                        "num_samples": len(base_logits_list),
                        "type": "multiple_choice",
                        "options_per_question": default_options
                    }
                    
                    print(f"[INFO] Final MCQ KL stats for {task_name}:")
                    print(f"       Mean: {mean_kl:.6f}")
                    print(f"       Min:  {min_kl:.6f}") 
                    print(f"       Max:  {max_kl:.6f}")
                    print(f"       Questions: {len(question_kls)}")
                    print(f"       Total options: {len(base_logits_list)}")
                    
                    return result
                else:
                    print(f"[ERROR] No valid question KL computations for {task_name}")
                    return {
                        "task_name": task_name,
                        "mean_kl": 0.0,
                        "num_questions": 0,
                        "type": "multiple_choice",
                        "error": "No valid question KL computations"
                    }
        
        else:  # Generative task
            print(f"[INFO] Computing generative KL divergence for {len(base_logits_list)} sample pairs")
            kl_values = []
            
            for i, (base_logits, new_logits) in enumerate(zip(base_logits_list, new_logits_list)):
                try:
                    kl = KLEval._kl_divergence(base_logits, new_logits)
                    kl_values.append(kl)
                    
                    if (i + 1) % 100 == 0:
                        current_mean = sum(kl_values) / len(kl_values)
                        print(f"[INFO] Processed {i + 1}/{len(base_logits_list)}, current mean KL: {current_mean:.6f}")
                        
                except Exception as e:
                    print(f"[WARN] KL computation error for sample {i}: {e}")
                    continue
            
            # Calculate final statistics
            if kl_values:
                mean_kl = sum(kl_values) / len(kl_values)
                min_kl = min(kl_values)
                max_kl = max(kl_values)
                
                result = {
                    "task_name": task_name,
                    "mean_kl": mean_kl,
                    "min_kl": min_kl,
                    "max_kl": max_kl,
                    "num_samples": len(kl_values),
                    "type": "generative"
                }
                
                print(f"[INFO] Final generative KL stats for {task_name}:")
                print(f"       Mean: {mean_kl:.6f}")
                print(f"       Min:  {min_kl:.6f}") 
                print(f"       Max:  {max_kl:.6f}")
                print(f"       Samples: {len(kl_values)}")
                
                return result
            else:
                print(f"[ERROR] No valid KL computations for {task_name}")
                return {
                    "task_name": task_name,
                    "mean_kl": 0.0,
                    "num_samples": 0,
                    "type": "generative",
                    "error": "No valid KL computations"
                }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute KL divergence between two model outputs (new manifest format).")
    parser.add_argument("base_model_dir", help="Directory with base model tensors")
    parser.add_argument("new_model_dir", help="Directory with new/quantized model tensors")
    parser.add_argument("--output", "-o", help="Output file prefix")
    parser.add_argument("--task", "-t", help="Optional substring filter for task names")
    args = parser.parse_args()

    kl = KLEval(args.base_model_dir, args.new_model_dir)
    kl.save_kl_results(args.output, task_filter=args.task)
