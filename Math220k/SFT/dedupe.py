import json
import argparse
import unicodedata
import re

def normalize_problem(p: str) -> str:
    p = unicodedata.normalize("NFKC", p)
    return re.sub(r"\s+", " ", p.strip())

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="Path to the JSONL file to deduplicate")
args = parser.parse_args()

seen = set()
output_lines = []

with open(args.file, "r") as f:
    for line in f:
        try:
            record = json.loads(line)
            prob = record.get("problem")
            if prob is None:
                continue
            norm = normalize_problem(prob)
            if norm not in seen:
                seen.add(norm)
                output_lines.append(line)
        except json.JSONDecodeError:
            continue

# Overwrite the original file
with open(args.file, "w") as f:
    for line in output_lines:
        f.write(line)

print(f"✅ Deduplicated: kept {len(output_lines)} unique problems in {args.file}")

