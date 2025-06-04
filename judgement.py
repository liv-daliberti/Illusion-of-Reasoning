#!/usr/bin/env python
import argparse
import json
import os
import glob
import tempfile
import time

print("Loading libraries…")
import torch
print("Torch version:", torch.__version__)

# AzureOpenAI is the client that can point at your AI Sandbox endpoint
from openai import AzureOpenAI
import requests
from datasets import load_dataset

# ──────────────────────────────────────────────────────────────────────────────
# Configuration: Use AI Sandbox chat endpoint as “DeepSeek”
# ──────────────────────────────────────────────────────────────────────────────
sandbox_api_key     = os.getenv("AI_SANDBOX_KEY", "08ccb8f51c534ebf9170337e15f01fef")
sandbox_endpoint    = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_version = "2025-03-01-preview"

# Pick the deployed model name that behaves like DeepSeek-R1.
DEEPSEEK_MODEL = "Meta-Llama-3-1-70B-Instruct-htzs"
MAX_JUDGE_TOKENS = 256  # truncate/pad prompts to this length

# Modified template: compare with baseline reasoning from earliest step
JUDGE_TEMPLATE = """\
You are a careful math grader. The model’s reasoning may differ, but focus on correctness and approach comparison.
First, read the ground-truth solution. Then read the baseline reasoning and final <answer> from the earliest checkpoint. Then read the current model’s reasoning and final <answer>.
Answer with exactly two tokens separated by a space:
• FIRST TOKEN  – “YES” or “NO” → Is the current model’s final answer correct?
• SECOND TOKEN – “YES” or “NO” → Does the current model’s approach differ from the baseline approach?
Do NOT write anything else.

Problem:
{problem}

Ground-truth solution:
{solution}

Baseline reasoning and final <answer>:
{baseline}

Current model output (chain-of-thought and final <answer>):
{current}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Send a ChatCompletion to the Sandbox, parse two-token reply
# ──────────────────────────────────────────────────────────────────────────────
def judge_with_sandbox(problem: str, solution: str, baseline_output: str, current_output: str):
    prompt_text = JUDGE_TEMPLATE.format(
        problem=problem.strip(),
        solution=solution.strip(),
        baseline=baseline_output.strip(),
        current=current_output.strip()
    )
    client = AzureOpenAI(
        api_key=sandbox_api_key,
        azure_endpoint=sandbox_endpoint,
        api_version=sandbox_api_version
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.0,
        max_tokens=8,
        messages=[
            {"role": "system",  "content": "You are a careful math grader."},
            {"role": "user",    "content": prompt_text},
        ],
    )
    reply = response.choices[0].message.content.strip().upper()
    parts = reply.split()
    if len(parts) >= 2 and parts[0] in {"YES", "NO"} and parts[1] in {"YES", "NO"}:
        is_correct = (parts[0] == "YES")
        changed   = (parts[1] == "YES")
        return is_correct, changed
    return False, False

# ──────────────────────────────────────────────────────────────────────────────
# Main: Build baseline map, then judge each record relative to baseline
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge inference JSONL files with relative approach change from earliest checkpoint."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing inference JSONL files (recursive)."
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="If set, redo all judgments even if flags already exist."
    )
    args = parser.parse_args()
    INPUT_ROOT = args.input_dir
    redo = args.redo

    # ──────────────────────────────────────────────────────────────────────────
    # Change #1: Instead of a single glob‐pattern, we keep a list of two patterns:
    #
    #   1) “**/*Instruct-SFT*_train.jsonl”  — matches SFT‐style files
    #   2) “**/*Instruct-GRPO*_train.jsonl” — matches GRPO‐style files
    #
    # We'll glob each pattern in turn and then dedupe+sort.
    #
    patterns = [
        "**/*Instruct-SFT*_train.jsonl",
        "**/*Instruct-GRPO*_train.jsonl"
    ]

    # Gather all matching files for both patterns
    all_files = []
    for pat in patterns:
        found = glob.glob(os.path.join(INPUT_ROOT, pat), recursive=True)
        all_files.extend(found)

    # Deduplicate and sort
    all_files = sorted(set(all_files))

    # Print out which files we actually found
    print(f"[INFO] Using INPUT_ROOT = {INPUT_ROOT}")
    for pat in patterns:
        print(f"[INFO]   Glob pattern: {os.path.join(INPUT_ROOT, pat)}")
    print(f"[INFO] Found {len(all_files)} total matching file(s):")
    for path in all_files:
        print(f"    {path}")
    # ──────────────────────────────────────────────────────────────────────────

    # 1) Load Math-220k dataset for ground-truth solutions
    print("[INFO] Loading Math dataset for ground-truth solutions…")
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        split="train",
        cache_dir="/n/fs/similarity/open-r1/datasets_cache",
    )
    solution_map = {ex["problem"]: ex["solution"] for ex in ds}
    print(f"[INFO] Found {len(solution_map)} entries in solution_map.")

    # 2) First pass: gather earliest reasoning per problem
    baseline_map = {}  # problem -> (earliest_step, baseline_output)
    for input_path in all_files:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                problem = rec.get("problem")
                step    = rec.get("step")
                output  = rec.get("output", "")
                if (problem is None) or (step is None):
                    continue
                # If this problem is new, or if this “step” is smaller than any seen so far:
                if (problem not in baseline_map) or (step < baseline_map[problem][0]):
                    baseline_map[problem] = (step, output)
    print(f"[INFO] Baseline map built for {len(baseline_map)} problems.")

    # 3) Second pass: judge each record using baseline
    for input_path in all_files:
        print(f"\n[INFO] Starting file: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_lines = [ln for ln in f if ln.strip()]
        total = len(raw_lines)
        print(f"  [INFO] Found {total} non-blank JSONL records.")
        records = [json.loads(line) for line in raw_lines]

        # Build a quick map of “problem + gold_answer” → record so we can update in place
        existing_map = {}
        for rec in records:
            key = json.dumps(
                {"problem": rec["problem"], "gold_answer": rec.get("gold_answer", "")},
                sort_keys=True
            )
            existing_map[key] = rec

        judged_count  = 0
        correct_count = 0
        changed_count = 0

        for idx, rec in enumerate(records, start=1):
            problem = rec.get("problem")
            step    = rec.get("step")
            if (problem is None) or (step is None):
                print(f"    [WARN] Record {idx} missing 'step' or 'problem'; skipping.")
                continue

            corr_key   = f"{step}_correct"
            change_key = f"{step}_changed"

            # If we’re not redoing AND both flags already exist, skip
            if redo or (corr_key not in rec) or (change_key not in rec):
                solution_text   = solution_map.get(problem, "")
                baseline_output = baseline_map.get(problem, (None, ""))[1]

                # If no ground truth solution OR no baseline, mark both as False
                if (not solution_text) or (not baseline_output):
                    rec[corr_key]   = False
                    rec[change_key] = False
                else:
                    current_output = rec.get("output", "")
                    print(f"    [DEBUG] Judging record {idx}/{total} (step {step}) relative to baseline…")
                    is_corr, is_changed = judge_with_sandbox(
                        problem, solution_text, baseline_output, current_output
                    )
                    rec[corr_key]   = is_corr
                    rec[change_key] = is_changed
                    print(f"    [DEBUG] Judgment for step {step}: correct={is_corr}, changed_relative={is_changed}")

            judged_count += 1
            if rec.get(corr_key):
                correct_count += 1
            if rec.get(change_key):
                changed_count += 1

            if (idx % 50 == 0) or (idx == total):
                accuracy    = correct_count / judged_count if judged_count else 0.0
                change_rate = changed_count / judged_count if judged_count else 0.0
                print(
                    f"    [PROGRESS] Processed {idx}/{total}, "
                    f"Accuracy: {correct_count}/{judged_count} ({accuracy:.2%}), "
                    f"Changed from baseline: {changed_count}/{judged_count} ({change_rate:.2%})"
                )

        # Write the updated records back to a temp file, then replace original
        dirpath, filename = os.path.split(input_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dirpath, delete=False) as tmpf:
            for rec in records:
                tmpf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            temp_path = tmpf.name

        os.replace(temp_path, input_path)

        final_acc         = correct_count / judged_count if judged_count else 0.0
        final_change_rate = changed_count / judged_count if judged_count else 0.0
        print(f"[INFO] Completed file: {input_path}")
        print(f"[INFO] Final accuracy: {correct_count}/{judged_count} ({final_acc:.2%})")
        print(f"[INFO] Final change-rate: {changed_count}/{judged_count} ({final_change_rate:.2%})")
