import os
import json
import argparse
from collections import defaultdict

def normalize_problem(p: str) -> str:
    return " ".join(p.strip().split())  # collapse all whitespace

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, help="Directory containing .jsonl files")
args = parser.parse_args()

directory = args.directory
filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jsonl")]

problem_to_lines = defaultdict(lambda: defaultdict(str))  # norm_problem -> file -> full line
file_to_problems = {}

for path in filepaths:
    problems = set()
    with open(path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                prob = record.get("problem")
                if not prob:
                    continue
                norm_prob = normalize_problem(prob)
                problems.add(norm_prob)
                problem_to_lines[norm_prob][path] = line
            except json.JSONDecodeError:
                continue
    file_to_problems[path] = problems

# Intersection of normalized problems
shared_norm_problems = set.intersection(*file_to_problems.values())
if len(shared_norm_problems) < 500:
    print(f"❗ Only {len(shared_norm_problems)} shared normalized problems found. Aborting.")
    exit(1)

# Preserve order from first file
first_file = filepaths[0]
ordered_shared_problems = []
with open(first_file, "r") as f:
    for line in f:
        try:
            record = json.loads(line)
            prob = record.get("problem")
            if prob:
                norm_prob = normalize_problem(prob)
                if norm_prob in shared_norm_problems:
                    ordered_shared_problems.append(norm_prob)
        except json.JSONDecodeError:
            continue
    ordered_shared_problems = ordered_shared_problems[:500]

# Truncate files
for path in filepaths:
    with open(path, "w") as out:
        for norm_prob in ordered_shared_problems:
            out.write(problem_to_lines[norm_prob][path])
    print(f"✅ Truncated: {os.path.basename(path)} to 500 shared problems")

print("🎯 Done.")
