import json


class FlipEval:
    """
    Compute flips (correct→wrong and wrong→correct) between two lm-eval-harness JSONL outputs.
    """

    def __init__(self, base_file_path: str, new_file_path: str):
        self.base_file_path = base_file_path
        self.new_file_path = new_file_path

        # Load raw JSON objects keyed by doc_id
        self.base = self._load_jsonl(self.base_file_path)
        self.new = self._load_jsonl(self.new_file_path)

        # Determine which binary signal to use ("acc" or "exact_match")
        self.signal_key = self._detect_signal_key(self.base)

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
        """
        first = next(iter(sample_map.values()))
        metrics = first.get("metrics", [])
        if "acc" in metrics:
            return "acc"
        if "exact_match" in metrics:
            return "exact_match"
        raise ValueError(f"No binary metric found in metrics: {metrics}")

    def count_flips(self, verbose: bool = False) -> dict[str, int | float | list]:
        """
        Compare base vs. new, assert same docs, and count flips.
        Returns a dict with:
          - total_examples
          - total_flips
          - percent_flips
          - correct_to_wrong
          - wrong_to_correct
          - correct_to_wrong_samples (if verbose=True)
          - wrong_to_correct_samples (if verbose=True)
        """
        correct_to_wrong = 0
        wrong_to_correct = 0
        matched = 0
        
        # For storing the actual flipped samples
        correct_to_wrong_samples = []
        wrong_to_correct_samples = []

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
                    if verbose:
                        correct_to_wrong_samples.append({
                            "doc_id": doc_id,
                            "query": base_obj["doc"]["query"] if "query" in base_obj["doc"] else "",
                            "choices": base_obj["doc"]["choices"] if "choices" in base_obj["doc"] else [],
                            "gold": base_obj["doc"]["gold"] if "gold" in base_obj["doc"] else "",
                            "gold_text": base_obj["doc"]["choices"][gold_idx] if gold_idx is not None and "choices" in base_obj["doc"] else "",
                            "base_prediction": base_pred,
                            "new_prediction": new_pred,
                            "base_scores": self._get_scores(base_obj),
                            "new_scores": self._get_scores(new_obj)
                        })
                else:
                    wrong_to_correct += 1
                    if verbose:
                        wrong_to_correct_samples.append({
                            "doc_id": doc_id,
                            "query": base_obj["doc"]["query"] if "query" in base_obj["doc"] else "",
                            "choices": base_obj["doc"]["choices"] if "choices" in base_obj["doc"] else [],
                            "gold": base_obj["doc"]["gold"] if "gold" in base_obj["doc"] else "",
                            "gold_text": base_obj["doc"]["choices"][gold_idx] if gold_idx is not None and "choices" in base_obj["doc"] else "",
                            "base_prediction": base_pred,
                            "new_prediction": new_pred,
                            "base_scores": self._get_scores(base_obj),
                            "new_scores": self._get_scores(new_obj)
                        })

        total_flips = correct_to_wrong + wrong_to_correct
        percent_flips = (total_flips / matched * 100) if matched else 0.0

        result = {
            "total_examples": matched,
            "total_flips": total_flips,
            "percent_flips": percent_flips,
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
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


if __name__ == "__main__":
    # Example usage:
    BASE_FILE_PATH = "output/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/samples_hellaswag_2025-05-21T00-44-07.577199.jsonl"
    NEW_FILE_PATH = "output/unsloth__DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit/samples_hellaswag_2025-05-21T00-47-34.834490.jsonl"
    fe = FlipEval(BASE_FILE_PATH, NEW_FILE_PATH)
    stats = fe.count_flips(verbose=True)

    print(f"Compared {stats['total_examples']} examples")
    print(f"Total flips: {stats['total_flips']} ({stats['percent_flips']:.2f}%)")
    print(f"  Correct → Wrong: {stats['correct_to_wrong']}")
    print(f"  Wrong → Correct: {stats['wrong_to_correct']}")
    
    if "correct_to_wrong_samples" in stats and stats["correct_to_wrong_samples"]:
        print("\nCorrect → Wrong Samples:")
        for i, sample in enumerate(stats["correct_to_wrong_samples"]):
            print(f"\nSample {i+1}:")
            print(f"  Doc ID: {sample['doc_id']}")
            print(f"  Query: {sample['query']}")
            print(f"  Gold answer: {sample['gold_text']}")
            
            print(f"  Base model (Correct):")
            print(f"    Prediction: Choice {sample['base_prediction']}")
            if sample['choices'] and sample['base_prediction'] is not None:
                print(f"    Text: {sample['choices'][sample['base_prediction']]}")
            
            print(f"  New model (Wrong):")
            print(f"    Prediction: Choice {sample['new_prediction']}")
            if sample['choices'] and sample['new_prediction'] is not None:
                print(f"    Text: {sample['choices'][sample['new_prediction']]}")
            
            print("  Scores:")
            for idx, (base_score, new_score) in enumerate(zip(sample['base_scores'] or [], sample['new_scores'] or [])):
                print(f"    Choice {idx}: Base={base_score[0]}, New={new_score[0]}")
            
    if "wrong_to_correct_samples" in stats and stats["wrong_to_correct_samples"]:
        print("\nWrong → Correct Samples:")
        for i, sample in enumerate(stats["wrong_to_correct_samples"]):
            print(f"\nSample {i+1}:")
            print(f"  Doc ID: {sample['doc_id']}")
            print(f"  Query: {sample['query']}")
            print(f"  Gold answer: {sample['gold_text']}")
            
            print(f"  Base model (Wrong):")
            print(f"    Prediction: Choice {sample['base_prediction']}")
            if sample['choices'] and sample['base_prediction'] is not None:
                print(f"    Text: {sample['choices'][sample['base_prediction']]}")
            
            print(f"  New model (Correct):")
            print(f"    Prediction: Choice {sample['new_prediction']}")
            if sample['choices'] and sample['new_prediction'] is not None:
                print(f"    Text: {sample['choices'][sample['new_prediction']]}")
            
            print("  Scores:")
            for idx, (base_score, new_score) in enumerate(zip(sample['base_scores'] or [], sample['new_scores'] or [])):
                print(f"    Choice {idx}: Base={base_score[0]}, New={new_score[0]}")
