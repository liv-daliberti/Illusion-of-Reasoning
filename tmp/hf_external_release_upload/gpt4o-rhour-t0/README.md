---
license: mit
language:
- en
tags:
- reasoning
- chain-of-thought
- external-models
- rhour
- gpt-4o 
size_categories:
- 1K<n<10K
---

# gpt4o-rhour-t0

This dataset repo contains external-model traces used in the Illusion-of-Reasoning analysis.

- Domain: `RHour`
- Model: `GPT-4o`
- Target temperature label: `0`
- Rows in `data.jsonl`: `3994`
- Source artifact: `/n/fs/similarity/Illusion-of-Reasoning/artifacts/results/gpt4o-carpark-azure-500x8`

Notes:
- Merged from the split local root, preferring records from step0000_test_500x8.jsonl on duplicate (example_id, sample_idx) keys. The resulting payload contains 3994 unique traces; the local artifact is short of the nominal 4000.
- Rows are uploaded as raw JSONL records from the local analysis artifacts.
