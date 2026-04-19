---
license: mit
language:
- en
tags:
- reasoning
- chain-of-thought
- external-models
- math-500
- deepseek-r1 
size_categories:
- 1K<n<10K
---

# deepseek-r1-math500-t0

This dataset repo contains external-model traces used in the Illusion-of-Reasoning analysis.

- Domain: `MATH-500`
- Model: `DeepSeek-R1`
- Target temperature label: `0`
- Rows in `data.jsonl`: `4000`
- Source artifact: `/n/fs/similarity/Illusion-of-Reasoning/artifacts/results/deepseek-r1-openrouter/step0000_test.jsonl`

Notes:
- Full local artifact root. This root contains a small number of records tagged with temperature=0.05 in-file.
- Rows are uploaded as raw JSONL records from the local analysis artifacts.
