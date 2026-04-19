---
language:
- en
pretty_name: Illusion of Reasoning - Main Traces
---

# Illusion of Reasoning - Main Traces

## Overview
This dataset contains reasoning traces from GRPO models across multiple tasks and temperatures. It includes:
- GRPO-1.5B on math, crosswords, and carpark (temps: 0, 0.05, 0.3, 0.7)
- GRPO-7B on math (temps: 0, 0.05, 0.3, 0.7)
- GRPO-Llama8B on math (temps: 0, 0.05, 0.3, 0.7)

The dataset contains a processed split with explicit metadata columns, plus the original raw JSONL files under
`artifacts/results/...` in the repository.

## Processed split
The processed dataset is created from `*_test.jsonl` trace files and adds explicit metadata columns:
- `model`: model name (e.g., `GRPO-1.5B`, `GRPO-7B`, `GRPO-Llama8B`)
- `temperature`: decoding temperature as a float
- `task`: task name (math, xword, carpark)
- `step`: training step (int)

To avoid schema conflicts across tasks, nested pass data are stored as JSON strings:
- `pass1`, `pass2`, `pass2a`, `pass2b`, `pass2c`

Other columns may be task-specific or optional; missing fields are set to null.

### Schema (core columns)
- `model`: string
- `temperature`: float
- `task`: string
- `step`: int
- `sample_idx`: int
- `split`: string
- `problem`: string (math)
- `example_id`: string (carpark)
- `enumeration`: string (xword)
- `gold_answer`: list of string
- `gold_answer_canon`: string
- `gold_answer_canon_set`: list of string
- `pass1`, `pass2`, `pass2a`, `pass2b`, `pass2c`: JSON-encoded strings

## Usage
```python
from datasets import load_dataset
import json

# Load the processed split
# (This loads the default config created by push_to_hub)
ds = load_dataset("od2961/illusion-of-reasoning-main-traces", split="train")

# Example: parse pass1 JSON
row = ds[0]
pass1 = json.loads(row["pass1"]) if row.get("pass1") else None
print(row["model"], row["temperature"], row["task"], row["step"])
```

## Notes
- Raw JSONL files are also available in the repository under `artifacts/results/...`.
- The processed split normalizes `gold_answer` and `gold_answer_canon_set` to lists of strings.

## License
No explicit license is declared in this dataset card. Please refer to the project repository for licensing details.
