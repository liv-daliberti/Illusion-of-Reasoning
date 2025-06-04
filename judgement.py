import argparse
import json
import os
import glob
import tempfile
import time
import base64
from mimetypes import guess_type

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
# Ensure you have set AI_SANDBOX_KEY in your environment:
#
#    export AI_SANDBOX_KEY=<your_sandbox_key>
#
sandbox_api_key     = "08ccb8f51c534ebf9170337e15f01fef"
sandbox_endpoint    = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_version = "2025-03-01-preview"

# Pick the deployed model name that behaves like DeepSeek-R1. For example:
DEEPSEEK_MODEL = "Meta-Llama-3-1-70B-Instruct-htzs"

MAX_JUDGE_TOKENS = 256  # we will truncate/pad prompts to this length

# Instruct “DeepSeek” (Sandbox) to return two tokens:
#  • FIRST  → “YES” or “NO”: final answer correct?
#  • SECOND → “YES” or “NO”: changed approach mid-way?
JUDGE_TEMPLATE = """\
You are a careful math grader. The model’s reasoning may differ from yours, but that is okay.
First, read the ground-truth solution. Then, read the model’s reasoning and final <answer>.
Answer with exactly two tokens separated by a space:
• FIRST TOKEN  – “YES” or “NO” → Is the model’s final answer correct?
• SECOND TOKEN – “YES” or “NO” → Did the model change approach or strategy part-way through?
Do NOT write anything else beyond those two tokens.

Problem:
{problem}

Ground-truth solution:
{solution}

Model output (including chain-of-thought and final <answer>):
{gen}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Send a ChatCompletion to the Sandbox, parse the “YES/NO” pair
# ──────────────────────────────────────────────────────────────────────────────
def judge_with_sandbox(problem: str, solution: str, gen_output: str):
    """
    Returns (is_correct: bool, changed_approach: bool)
    by calling the AI Sandbox chat completion with the JUDGE_TEMPLATE.
    """

    # 1) Fill in the template
    prompt_text = JUDGE_TEMPLATE.format(
        problem=problem.strip(),
        solution=solution.strip(),
        gen=gen_output.strip()
    )

    # 2) Build the OpenAI chat payload
    client = AzureOpenAI(
        api_key=sandbox_api_key,
        azure_endpoint=sandbox_endpoint,
        api_version=sandbox_api_version
    )

    # 3) Send the chat completion request
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.0,       # deterministic
        max_tokens=8,          # only need two tokens (“YES NO”)
        messages=[
            {"role": "system",  "content": "You are a careful math grader."},
            {"role": "user",    "content": prompt_text},
        ],
    )

    # 4) Parse out the two‐token reply
    reply = response.choices[0].message.content.strip().upper()
    parts = reply.split()
    if len(parts) >= 2 and parts[0] in {"YES", "NO"} and parts[1] in {"YES", "NO"}:
        is_correct = (parts[0] == "YES")
        changed   = (parts[1] == "YES")
        return is_correct, changed

    # If something unexpected was returned, default to False, False:
    return False, False


# ──────────────────────────────────────────────────────────────────────────────
# Main: Read inference JSONL files, append “_correct” and “_changed” flags
#        by using judge_with_sandbox() instead of a local model
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge inference JSONL files via AI Sandbox (DeepSeek-R1 style)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing inference JSONL files (recursive)."
    )
    args = parser.parse_args()

    # 1) Load the Math-220k dataset for ground-truth solutions
    print("[INFO] Loading Math dataset for ground-truth solutions…")
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        split="train",
        cache_dir="/n/fs/similarity/open-r1/datasets_cache",
    )
    solution_map = {ex["problem"]: ex["solution"] for ex in ds}
    print(f"[INFO] Found {len(solution_map)} entries in solution_map.")

    INPUT_ROOT = args.input_dir
    PATTERN = "**/Qwen2.5-1.5B-Instruct-*train.jsonl"

    for input_path in sorted(glob.glob(os.path.join(INPUT_ROOT, PATTERN), recursive=True)):
        print(f"\n[INFO] Starting file: {input_path}")

        # 2) Read existing lines (skip blank lines)
        with open(input_path, "r", encoding="utf-8") as f:
            raw_lines = [ln for ln in f if ln.strip()]
        total = len(raw_lines)
        print(f"  [INFO] Found {total} non-blank JSONL records.")

        # 3) Parse all records
        records = [json.loads(line) for line in raw_lines]

        # 4) Build a lookup map by (problem, gold_answer) to preserve prior flags
        existing_map = {}
        for rec in records:
            key = json.dumps(
                {"problem": rec["problem"], "gold_answer": rec["gold_answer"]},
                sort_keys=True
            )
            existing_map[key] = rec

        judged_count = 0
        correct_count = 0
        changed_count = 0

        # 5) Iterate and judge each record if not already judged
        for idx, rec in enumerate(records, start=1):
            key = json.dumps(
                {"problem": rec["problem"], "gold_answer": rec["gold_answer"]},
                sort_keys=True
            )
            rec.update(existing_map.get(key, {}))

            rev = rec.get("step")
            if not rev:
                print(f"    [WARN] Record {idx} missing 'step'; skipping.")
                continue

            corr_key   = f"{rev}_correct"
            change_key = f"{rev}_changed"

            # Only call the sandbox if we haven’t already stored these flags
            if corr_key not in rec or change_key not in rec:
                problem_text  = rec["problem"]
                solution_text = solution_map.get(problem_text, "")
                if not solution_text:
                    print(f"    [WARN] No solution found for problem {idx}; skipping judgment.")
                    rec[corr_key]   = False
                    rec[change_key] = False
                else:
                    gen_output = rec.get("output", "")
                    print(f"    [DEBUG] Judging record {idx}/{total} (step {rev}) via Sandbox…")
                    is_corr, is_changed = judge_with_sandbox(
                        problem_text, solution_text, gen_output
                    )
                    rec[corr_key]   = is_corr
                    rec[change_key] = is_changed
                    print(f"    [DEBUG] Judgment for step {rev}: correct={is_corr}, changed_approach={is_changed}")

            judged_count += 1
            if rec[corr_key]:
                correct_count += 1
            if rec[change_key]:
                changed_count += 1

            # Print progress every 50 or at the end
            if idx % 50 == 0 or idx == total:
                accuracy    = correct_count / judged_count if judged_count else 0.0
                change_rate = changed_count / judged_count  if judged_count else 0.0
                print(
                    f"    [PROGRESS] Processed {idx}/{total}, "
                    f"Accuracy: {correct_count}/{judged_count} ({accuracy:.2%}), "
                    f"Changed Approach: {changed_count}/{judged_count} ({change_rate:.2%})"
                )

        # 6) Write back to a temp file, then replace the original JSONL
        dirpath, filename = os.path.split(input_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dirpath, delete=False) as tmpf:
            for rec in records:
                tmpf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            temp_path = tmpf.name

        os.replace(temp_path, input_path)
        final_acc        = correct_count / judged_count    if judged_count else 0.0
        final_change_rate = changed_count / judged_count   if judged_count else 0.0
        print(f"[INFO] Completed file: {input_path}")
        print(f"[INFO] Final accuracy: {correct_count}/{judged_count} ({final_acc:.2%})")
        print(f"[INFO] Final change-rate: {changed_count}/{judged_count} ({final_change_rate:.2%})")
