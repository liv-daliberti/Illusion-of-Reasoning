#!/usr/bin/env python
import os
import json
import re
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute accuracy for each GRPO checkpoint file and save a plot.")
parser.add_argument(
    "directory",
    type=str,
    help="Path to the directory containing .jsonl files like Qwen2.5-1.5B-Instruct-GRPO_step_XXXX_train.jsonl",
)
args = parser.parse_args()
directory = args.directory

# Pattern to match GRPO filenames and extract step number
pattern = re.compile(r"Qwen2\.5-1\.5B-Instruct-GRPO_step_(\d+)_train\.jsonl")

# Dictionary to hold step -> accuracy
step_to_accuracy = {}

# Scan and process files in the given directory
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if not match:
        continue

    step = int(match.group(1))
    filepath = os.path.join(directory, filename)

    correct = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Look for exactly one key ending in "_correct" per record
            for k, v in record.items():
                if k.endswith("_correct"):
                    total += 1
                    if v is True:
                        correct += 1
                    break  # there should be only one "_correct" key per record

    step_to_accuracy[step] = (correct / total) if total > 0 else None

# Prepare data for plotting
steps = sorted(step_to_accuracy.keys())
accuracies = [step_to_accuracy[s] for s in steps]

# Filter out steps with no data
filtered_steps = [s for s, a in zip(steps, accuracies) if a is not None]
filtered_accuracies = [a for a in accuracies if a is not None]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(filtered_steps, filtered_accuracies, marker='o')
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("GRPO Checkpoint Accuracy over Steps")
plt.grid(True)
plt.tight_layout()

# Save the plot in the same directory
output_path = os.path.join(directory, "accuracy_plot.png")
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")