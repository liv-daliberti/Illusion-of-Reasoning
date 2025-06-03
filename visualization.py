import os
import json
import re
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute accuracy for each checkpoint file.")
parser.add_argument(
    "directory",
    type=str,
    help="Path to the directory containing .jsonl files like Qwen2.5-1.5B-Instruct-SFT_stepXXXX_train.jsonl",
)
args = parser.parse_args()
directory = args.directory

# Pattern to match filename and extract step number
pattern = re.compile(r"Qwen2\.5-1\.5B-Instruct-SFT_step(\d+)_train\.jsonl")

# Dictionary to hold step -> accuracy
step_to_accuracy = {}

# Scan and process files
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if not match:
        continue
    step = int(match.group(1))
    filepath = os.path.join(directory, filename)

    correct = 0
    total = 0

    with open(filepath, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            for k, v in record.items():
                if k.endswith("_correct"):
                    total += 1
                    if v is True:
                        correct += 1
                    break  # Only one "_correct" key per record

    step_to_accuracy[step] = correct / total if total > 0 else None

# Output results
print("Step\tAccuracy")
for step in sorted(step_to_accuracy.keys()):
    acc = step_to_accuracy[step]
    if acc is None:
        print(f"{step}\tNo data")
    else:
        print(f"{step}\t{acc:.4f}")
