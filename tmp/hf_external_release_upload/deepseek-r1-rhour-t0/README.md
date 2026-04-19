---
license: mit
language:
- en
tags:
- reasoning
- chain-of-thought
- external-models
- rhour
- deepseek-r1 
size_categories:
- 1K<n<10K
---

# deepseek-r1-rhour-t0

This dataset repo contains external-model traces used in the Illusion-of-Reasoning analysis.

- Domain: `RHour`
- Model: `DeepSeek-R1`
- Target temperature label: `0`
- Rows in `data.jsonl`: `4000`
- Source artifact: `/n/fs/similarity/Illusion-of-Reasoning/artifacts/results/deepseek-r1-carpark-azure-500x8/step0000_test.jsonl`

Notes:
- Full local artifact root. This root contains mixed in-file temperature values (0, 0.05, 0.3, 0.7, 1.0).
- Rows are uploaded as raw JSONL records from the local analysis artifacts.
