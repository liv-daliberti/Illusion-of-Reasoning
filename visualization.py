#!/usr/bin/env python
import os
import json
import re
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute accuracy and 'changed' rate for each GRPO checkpoint file and save separate plots.")
parser.add_argument(
    "directory",
    type=str,
    help="Path to the directory containing .jsonl files like Qwen2.5-1.5B-Instruct-GRPO_step_XXXX_train.jsonl",
)
args = parser.parse_args()
directory = args.directory

# Pattern to match GRPO filenames and extract step number
pattern = re.compile(r"Qwen2\.5-1\.5B-Instruct-GRPO_step_(\d+)_train\.jsonl")

# Dictionaries to hold step -> accuracy and step -> changed rate
step_to_accuracy = {}
step_to_changed = {}

# Scan and process files in the given directory
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if not match:
        continue

    step = int(match.group(1))
    filepath = os.path.join(directory, filename)

    correct = 0
    changed_true = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Variables to store this record's flags
            this_correct = None
            this_changed = None

            # Look for keys ending in "_correct" and "_changed"
            for k, v in record.items():
                if k.endswith("_correct"):
                    this_correct = v
                elif k.endswith("_changed"):
                    this_changed = v

            # Only count if we found a "_correct" entry
            if this_correct is not None:
                total += 1
                if this_correct is True:
                    correct += 1
                if this_changed is True:
                    changed_true += 1

    # Compute ratios if total > 0
    step_to_accuracy[step] = (correct / total) if total > 0 else None
    step_to_changed[step] = (changed_true / total) if total > 0 else None

# Prepare data for plotting: only include steps with valid data
steps = sorted(step_to_accuracy.keys())
accuracies = [step_to_accuracy[s] for s in steps]
changed_rates = [step_to_changed[s] for s in steps]

filtered_steps = [s for s, a in zip(steps, accuracies) if a is not None and step_to_changed[s] is not None]
filtered_accuracies = [step_to_accuracy[s] for s in filtered_steps]
filtered_changed = [step_to_changed[s] for s in filtered_steps]

# Plot 1: Accuracy over Steps
plt.figure(figsize=(10, 6))
plt.plot(filtered_steps, filtered_accuracies, marker='o')
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("GRPO Checkpoint Accuracy over Steps")
plt.grid(True)
plt.tight_layout()
accuracy_output = os.path.join(directory, "accuracy_plot.png")
plt.savefig(accuracy_output)
print(f"Accuracy plot saved to: {accuracy_output}")
plt.close()

# Plot 2: Changed Rate over Steps
plt.figure(figsize=(10, 6))
plt.plot(filtered_steps, filtered_changed, marker='x', color='orange')
plt.xlabel("Step")
plt.ylabel("Changed Rate")
plt.title("GRPO Checkpoint Changed Rate over Steps")
plt.grid(True)
plt.tight_layout()
changed_output = os.path.join(directory, "changed_rate_plot.png")
plt.savefig(changed_output)
print(f"Changed rate plot saved to: {changed_output}")
plt.close()