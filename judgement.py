import argparse
import json
import os
import glob
import tempfile
import time
from tqdm import tqdm
import openai
from openai import OpenAI

# ——— helper to ask ChatGPT (v1 API) with rate-limit handling ——————————————————
def judge_answer(gen_output: str, ground_truth: str, max_retries=20, backoff=2) -> bool:
    prompt = f"""
Ground truth: {ground_truth}

Model output:
{gen_output}

Question: Is the model’s <answer> correct? Reply “yes” or “no” only.
"""
    retries = 0
    while True:
        try:
            print("    [DEBUG] Sending API request...")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            print("    [DEBUG] Received API response.")
            return response.choices[0].message.content.strip().lower().startswith("y")
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "tpm" in err_str:
                retries += 1
                if retries > max_retries:
                    print("    [ERROR] Max retries reached, re-raising exception.")
                    raise
                wait_time = backoff ** retries
                print(f"    [WARN] Rate limit hit. Sleeping for {wait_time}s... (retry {retries}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"    [ERROR] Unexpected exception: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="In-place judge of inference JSONL files under a chosen directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory containing inference JSONL files (recursive search)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not set via OPENAI_API_KEY environment variable)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Chat model name to use for judgments (default: gpt-4o)."
    )
    args = parser.parse_args()

    # ——— CONFIG —————————————————————————————————————————————
    api_key_used = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key_used:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY env var.")
    client = OpenAI(api_key=api_key_used)
    CHAT_MODEL = args.model

    print(f"[INFO] Using model: {CHAT_MODEL}")
    print(f"[INFO] Input directory: {args.input_dir}")

    # ——— PARAMETERS ———————————————————————————————————————————
    INPUT_ROOT = args.input_dir
    PATTERN = "**/Qwen2.5-1.5B-Instruct-SFT_step*_train.jsonl"

    # ——— MAIN ——————————————————————————————————————————————————
    for input_path in sorted(glob.glob(os.path.join(INPUT_ROOT, PATTERN), recursive=True)):
        print(f"\n[INFO] Starting file: {input_path}")
        # 1. Read existing lines
        print("  [DEBUG] Reading file...")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        # Filter out any blank lines
        lines = [ln for ln in raw_lines if ln.strip()]
        total = len(lines)
        print(f"  [DEBUG] Total non-blank records found: {total}")

        # 2. Parse all records
        print("  [DEBUG] Parsing JSON lines into records...")
        records = [json.loads(line) for line in lines]

        # 3. Build a map from (problem, gold_answer) → record
        print("  [DEBUG] Building lookup map for existing records...")
        existing_map = {}
        for rec in records:
            key = json.dumps(
                {"problem": rec.get("problem"), "gold_answer": rec.get("gold_answer")},
                sort_keys=True
            )
            existing_map[key] = rec
        print(f"  [DEBUG] Lookup map size: {len(existing_map)}")

        # Counters for running accuracy
        judged_count = 0
        correct_count = 0

        # 4. Update records in place
        print("  [INFO] Beginning to judge individual records...")
        for idx, rec in enumerate(records, start=1):
            key = json.dumps(
                {"problem": rec.get("problem"), "gold_answer": rec.get("gold_answer")},
                sort_keys=True
            )
            base_rec = existing_map.get(key, {})
            rec.update(base_rec)

            rev = rec.get("step")
            if not rev:
                print(f"    [WARN] Record {idx} missing 'step'; skipping.")
                continue
            judgment_key = f"{rev}_correct"

            if judgment_key not in rec:
                print(f"    [DEBUG] Judging record {idx}/{total} (step {rev})...")
                ground = rec["gold_answer"]
                gen = rec.get("output", "")
                rec[judgment_key] = judge_answer(gen, ground)
                print(f"    [DEBUG] Judgment for step {rev}: {rec[judgment_key]}")

            # Now a judgment exists
            judged_count += 1
            if rec[judgment_key]:
                correct_count += 1

            # Print running accuracy every 50 examples and at the end
            if idx % 50 == 0 or idx == total:
                accuracy = correct_count / judged_count if judged_count > 0 else 0.0
                print(f"    [PROGRESS] Processed {idx}/{total}, "
                      f"Accuracy so far: {correct_count}/{judged_count} ({accuracy:.2%})")

        # 5. Write updated records back to a temp file, then replace original
        print("  [DEBUG] Writing updated records to temporary file...")
        dirpath, filename = os.path.split(input_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dirpath, delete=False) as tmpf:
            for rec in records:
                tmpf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            temp_path = tmpf.name
        print("  [DEBUG] Replacing original file with updated file...")
        os.replace(temp_path, input_path)

        final_accuracy = correct_count / judged_count if judged_count > 0 else 0.0
        print(f"[INFO] Completed file: {input_path}")
        print(f"[INFO] Final accuracy: {correct_count}/{judged_count} ({final_accuracy:.2%})")
