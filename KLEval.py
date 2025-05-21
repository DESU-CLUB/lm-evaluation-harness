import torch

class KLEval:
    def __init__(self, base_tensor_path, new_tensor_path):
        self.base_tensor_path = base_tensor_path
        self.new_tensor_path = new_tensor_path
        self.base_tensors = torch.load(base_tensor_path)
        self.new_tensors = torch.load(new_tensor_path)

    @torch.no_grad()
    def calculate_per_task_kl(self):
        """Calculate KL divergence per task, handling MCQ tasks differently"""
        results = {}
        
        # Ensure both tensor sets have the same structure
        if len(self.base_tensors) != len(self.new_tensors):
            raise ValueError(f"Tensor sets have different lengths: {len(self.base_tensors)} vs {len(self.new_tensors)}")
        
        # Process each task
        for base_task_data, new_task_data in zip(self.base_tensors, self.new_tensors):
            print(base_task_data)
            task_name = base_task_data["task_name"]
            task_type = base_task_data["task_type"]  # We'll add this in evaluator.py
            if task_name != new_task_data["task_name"]:
                print(f"Warning: Task name mismatch: {task_name} vs {new_task_data['task_name']}")
                continue
                
            base_task_tensors = base_task_data["tensors"]
            new_task_tensors = new_task_data["tensors"]
            
            if len(base_task_tensors) != len(new_task_tensors):
                print(f"Warning: Different number of samples for task {task_name}")
                continue
                
            # For MCQ tasks, we need to know the number of options per question
            if task_type == "multiple_choice":
                # This will track KL per question
                question_kls = []
                current_question_kls = []
                options_per_question = base_task_data.get("options_per_question", [])
                
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
                        if base_sample.shape != new_sample.shape:
                            print(f"Skipping option with mismatched shape in question {question_idx}")
                            option_index += 1
                            continue
                        
                        # Convert to log probabilities and calculate per-token KL
                        base_log_probs = base_sample.log_softmax(dim=-1)
                        new_log_probs = new_sample.log_softmax(dim=-1)
                        token_kls = torch.nn.functional.kl_div(
                            new_log_probs,
                            base_log_probs.exp(),
                            reduction='none'
                        ).mean(dim=-1)  # Average across vocabulary
                        
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
                    "per_question_kls": question_kls
                }
            else:
                # For generative tasks - same as before
                sample_kls = []
                for base_sample, new_sample in zip(base_task_tensors, new_task_tensors):
                    if base_sample.shape != new_sample.shape:
                        print(f"Skipping sample with mismatched shapes in task {task_name}")
                        continue
                    
                    # Convert to log probabilities
                    base_log_probs = base_sample.log_softmax(dim=-1)
                    new_log_probs = new_sample.log_softmax(dim=-1)
                    
                    # Mean KL per token first
                    token_kls = torch.nn.functional.kl_div(
                        new_log_probs,
                        base_log_probs.exp(),
                        reduction='none'
                    ).mean(dim=-1)  # Average across vocabulary
                    
                    # Mean across sequence
                    sample_kl = token_kls.mean().item()
                    sample_kls.append(sample_kl)
                
                # Task-level KL is mean of all sample KLs
                task_kl = sum(sample_kls) / max(1, len(sample_kls))
                results[task_name] = {
                    "mean_kl": task_kl,
                    "type": "generative",
                    "num_samples": len(sample_kls),
                    "per_sample_kls": sample_kls
                }
        
        return results
        
if __name__ == "__main__":
    base_tensor_path = "DeepSeek-R1-Distill-Qwen-1.5B_tensors_by_task.pt"
    new_tensor_path = "DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit_tensors_by_task.pt"
    kl_eval = KLEval(base_tensor_path, new_tensor_path)
    
    # Overall KL
    print(f"Overall KL: {kl_eval.calculate_per_task_kl()}")
    

        
        