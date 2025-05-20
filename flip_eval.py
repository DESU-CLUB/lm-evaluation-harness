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
        self.new  = self._load_jsonl(self.new_file_path)

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

    def count_flips(self) -> dict[str, int | float]:
        """
        Compare base vs. new, assert same docs, and count flips.
        Returns a dict with:
          - total_examples
          - total_flips
          - percent_flips
          - correct_to_wrong
          - wrong_to_correct
        """
        correct_to_wrong = 0
        wrong_to_correct = 0
        matched = 0

        for doc_id, base_obj in self.base.items():
            # Ensure the same doc_id exists in new run
            assert doc_id in self.new, f"doc_id {doc_id} missing in new file"
            new_obj = self.new[doc_id]

            # Instance-level assert: ensure same document by comparing doc_hash
            base_hash = base_obj.get("doc_hash")
            new_hash  = new_obj.get("doc_hash")
            assert base_hash == new_hash, (
                f"doc_hash mismatch for doc_id {doc_id}: "
                f"{base_hash} (base) != {new_hash} (new)"
            )

            # Extract binary signals
            b = int(base_obj[self.signal_key])
            n = int(new_obj[self.signal_key])

            matched += 1
            if b != n:
                if b == 1 and n == 0:
                    correct_to_wrong += 1
                else:
                    wrong_to_correct += 1

        total_flips = correct_to_wrong + wrong_to_correct
        percent_flips = (total_flips / matched * 100) if matched else 0.0

        return {
            "total_examples": matched,
            "total_flips": total_flips,
            "percent_flips": percent_flips,
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
        }


if __name__ == "__main__":
    # Example usage:
    BASE_FILE_PATH = "/Users/warrenlow/Documents/Menlo/Inference/lm-evaluation-harness/output/gpt2/gpt2/samples_hellaswag_2025-05-20T12-55-21.381080.jsonl"
    NEW_FILE_PATH  = "/Users/warrenlow/Documents/Menlo/Inference/lm-evaluation-harness/output/gpt2/gpt2/samples_hellaswag_2025-05-20T17-51-27.109727.jsonl"

    fe = FlipEval(BASE_FILE_PATH, NEW_FILE_PATH)
    stats = fe.count_flips()

    print(f"Compared {stats['total_examples']} examples")
    print(f"Total flips: {stats['total_flips']} ({stats['percent_flips']:.2f}%)")
    print(f"  Correct → Wrong: {stats['correct_to_wrong']}")
    print(f"  Wrong → Correct: {stats['wrong_to_correct']}")
