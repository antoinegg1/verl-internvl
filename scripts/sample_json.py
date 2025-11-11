import sys
import json
import random
from pathlib import Path

def sample_jsonl_files(output_path: str, input_paths: list[str], ratio: float = 0.1) -> None:
    all_samples = []
    for p in input_paths:
        path = Path(p)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        n = int(len(lines) * ratio)
        if n < 1 and len(lines) > 0:
            n = 1
        if n > len(lines):
            n = len(lines)
        chosen = random.sample(lines, n)
        all_samples.extend(chosen)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in all_samples:
            line = line.rstrip("\n")
            if not line:
                continue
            out_f.write(line + "\n")

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("usage: python sample_jsonl.py output.jsonl input1.jsonl [input2.jsonl ...]")
    #     sys.exit(1)
    output_file = "/storage/openpsi/data/grounding_sft_v1/refcoco_test_sample.jsonl"
    input_files = ["/storage/openpsi/data/grounding_sft_v1/refcoco_testA.jsonl", "/storage/openpsi/data/grounding_sft_v1/refcoco_testB.jsonl", "/storage/openpsi/data/grounding_sft_v1/refcoco+_testA.jsonl", "/storage/openpsi/data/grounding_sft_v1/refcoco+_testB.jsonl", "/storage/openpsi/data/grounding_sft_v1/refcoco+_val.jsonl", "/storage/openpsi/data/grounding_sft_v1/refcocog_val.jsonl", "/storage/openpsi/data/grounding_sft_v1/refcocog_test.jsonl", "/storage/openpsi/data/grounding_sft_v1/refcoco_val.jsonl"]
    sample_jsonl_files(output_file, input_files, ratio=0.1)
