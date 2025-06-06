#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annotate_pivots.py
──────────────────
Adds per-checkpoint flags to inference JSONL files:

  • <step>_correct  – DeepSeek-graded correctness
  • <step>_changed  – pivot flag (True at every approach-shift segment
                      that eventually yields a correct answer)

Verbose logging shows exactly what’s happening at each step.

Usage
=====
python annotate_pivots.py \
    --input_dir Math220k/GRPO \
    --log-level INFO        # or DEBUG / WARNING
    [--redo]                # force recompute even if flags exist
"""
# ───────────────────────────── Imports ───────────────────────────────────────
import argparse, json, os, glob, tempfile, sys, time, logging
from contextlib import contextmanager

print("Loading libraries…")
import torch
print("Torch version:", torch.__version__)

from openai import AzureOpenAI
from datasets import load_dataset

# ─────────────────────── Helper: timing context ──────────────────────────────
@contextmanager
def timed(label: str):
    start = time.time()
    yield
    dur = time.time() - start
    logging.info(f"[TIMING] {label} — {dur:,.2f} s")

# ─────────────────── DeepSeek (AI-Sandbox) config ────────────────────────────
sandbox_api_key  = os.getenv("AI_SANDBOX_KEY",
                             "08ccb8f51c534ebf9170337e15f01fef")
sandbox_endpoint = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_ver  = "2025-03-01-preview"
DEEPSEEK_MODEL   = "Meta-Llama-3-1-70B-Instruct-htzs"
MAX_TOK_COR, MAX_TOK_BOTH = 4, 4

# ───────────────────── Prompt templates (unchanged) ──────────────────────────
CORRECT_ONLY_TMPL = """\
You are a careful math grader.

Problem:
{problem}

Ground-truth solution:
{solution}

Model output (chain-of-thought + <answer>):
{current}

Question: Is the model’s final <answer> correct?
Answer with exactly one token: “YES” or “NO”.  Do NOT write anything else.
"""

BOTH_TMPL = """\
You are a careful math grader. A model has two consecutive attempts to solve the same problem.

Problem:
{problem}

Ground-truth solution:
{solution}

PREVIOUS attempt (chain-of-thought + <answer>):
{previous}

CURRENT attempt (chain-of-thought + <answer>):
{current}

Answer with exactly two tokens separated by a space:
• FIRST TOKEN  – “YES” or “NO” → Is the CURRENT model’s final <answer> correct?
• SECOND TOKEN – “YES” or “NO” → Did the CURRENT model’s approach CHANGE relative to the PREVIOUS attempt?

Do NOT write anything else.
"""

# ───────────────────── DeepSeek wrapper functions ────────────────────────────
def _ds_chat(prompt: str, max_tok: int) -> str:
    client = AzureOpenAI(api_key=sandbox_api_key,
                         azure_endpoint=sandbox_endpoint,
                         api_version=sandbox_api_ver)
    with timed("DeepSeek call"):
        resp = client.chat.completions.create(
            model       = DEEPSEEK_MODEL,
            temperature = 0.0,
            max_tokens  = max_tok,
            messages    = [
                {"role": "system", "content": "You are a careful math grader."},
                {"role": "user",   "content": prompt},
            ],
        )
    answer = resp.choices[0].message.content.strip().upper()
    logging.debug(f"DeepSeek reply: {answer}")
    return answer

def check_correct_only(prob, sol, cur) -> bool:
    return _ds_chat(
        CORRECT_ONLY_TMPL.format(
            problem=prob.strip(),
            solution=sol.strip(),
            current=cur.strip()
        ), MAX_TOK_COR
    ) == "YES"

def check_both(prob, sol, prev, cur):
    reply = _ds_chat(
        BOTH_TMPL.format(
            problem=prob.strip(),
            solution=sol.strip(),
            previous=prev.strip(),
            current=cur.strip()
        ), MAX_TOK_BOTH
    ).split()
    if len(reply) >= 2 and reply[0] in {"YES","NO"} and reply[1] in {"YES","NO"}:
        return reply[0] == "YES", reply[1] == "YES"
    logging.warning(f"Malformed DeepSeek reply: {reply}")
    return False, False

# ─────────────────────── Multi-pivot detector ────────────────────────────────
def mark_pivots(seq):
    """
    seq: list[dict] with keys raw_changed, new_correct
    Adds final_changed flags:
      pivot = segment-start where raw_changed True AND segment contains correct
    """
    for e in seq:
        e["final_changed"] = False

    i = 0
    while i < len(seq):
        seg_start  = i
        start_flag = seq[i]["raw_changed"]
        i += 1
        while i < len(seq) and not seq[i]["raw_changed"]:
            i += 1
        seg_end = i  # exclusive

        if start_flag and any(r["new_correct"] for r in seq[seg_start:seg_end]):
            seq[seg_start]["final_changed"] = True
            logging.debug(f"Pivot at step {seq[seg_start]['step']}")
    return seq

# ────────────────────────────────── Main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Add correctness + pivot flags to JSONL inference files."
    )
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing \*_train.jsonl inference files.")
    ap.add_argument("--redo", action="store_true",
                    help="Recompute flags even if they already exist.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                    help="Set logging verbosity (default: INFO)")
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.info(f"Log level set to {args.log_level}")

    # ─── 1) Find JSONL files ending in `_train.jsonl` ────────────────────────────
    with timed("glob inference files"):
        files = sorted({p for p in glob.glob(
            os.path.join(args.input_dir, "**/*_train.jsonl"),
            recursive=True
        )})
    if not files:
        logging.error(f"No JSONL files under {args.input_dir}")
        sys.exit(1)
    logging.info(f"Found {len(files)} file(s) under {args.input_dir}")

    # ─── 2) Load each JSONL, bucket entries by problem ──────────────────────────
    file_recs, buckets = {}, {}
    with timed("load JSONL files"):
        for path in files:
            with open(path, encoding="utf-8") as fh:
                recs = [json.loads(l) for l in fh if l.strip()]
            file_recs[path] = recs

            for idx, r in enumerate(recs):
                prob, step = r.get("problem"), r.get("step")
                if prob is None or step is None:
                    continue
                buckets.setdefault(prob, []).append(dict(
                    file=path,
                    idx=idx,
                    step=step,
                    output=r.get("output",""),
                    new_correct=None,
                    raw_changed=None,
                    final_changed=None
                ))
    for lst in buckets.values():
        lst.sort(key=lambda e: e["step"])
    logging.info(f"Loaded {len(buckets)} distinct problems")

    # ─── 3) Load Math-220k ground-truth solutions ───────────────────────────────
    logging.info("Loading Math-220k ground-truth …")
    with timed("load_dataset"):
        ds = load_dataset("open-r1/OpenR1-Math-220k", split="train",
                          cache_dir="/n/fs/similarity/open-r1/datasets_cache")
    gt = {d["problem"]: d["solution"] for d in ds}
    logging.info(f"Ground-truth size: {len(gt):,}")

    # ─── 4) DeepSeek grading for correctness & raw_changed ──────────────────────
    for prob, seq in buckets.items():
        logging.debug(f"Processing problem (steps={len(seq)})")
        sol = gt.get(prob, "")
        for i, e in enumerate(seq):
            step = e["step"]
            rec  = file_recs[e["file"]][e["idx"]]
            corr_k, chg_k = f"{step}_correct", f"{step}_changed"

            if (not args.redo) and corr_k in rec and chg_k in rec:
                e["new_correct"] = bool(rec[corr_k])
                e["raw_changed"] = bool(rec[chg_k])
                logging.debug(
                    f"Reused flags for step {step}: "
                    f"correct={e['new_correct']} changed={e['raw_changed']}"
                )
                continue

            if not sol:
                e["new_correct"] = e["raw_changed"] = False
            elif i == 0:
                e["new_correct"] = check_correct_only(prob, sol, e["output"])
                e["raw_changed"] = False
            else:
                prev_out = seq[i-1]["output"]
                e["new_correct"], e["raw_changed"] = \
                    check_both(prob, sol, prev_out, e["output"])

            logging.debug(
                f"Step {step}: correct={e['new_correct']} raw_changed={e['raw_changed']}"
            )

        # ─── 5) Mark all pivots in this problem ─────────────────────────────────
        mark_pivots(seq)

    # ─── 6) Write all modified records back to disk ─────────────────────────────
    with timed("rewrite JSONL files"):
        for path, recs in file_recs.items():
            tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8",
                                              dir=os.path.dirname(path),
                                              delete=False)
            for idx, rec in enumerate(recs):
                prob, step = rec.get("problem"), rec.get("step")
                if prob in buckets and step is not None:
                    # lookup the bucket entry for this file+index
                    e = next(x for x in buckets[prob]
                             if x["file"] == path and x["idx"] == idx)
                    rec[f"{step}_correct"] = e["new_correct"]
                    rec[f"{step}_changed"] = e["final_changed"]
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.flush()
            os.replace(tmp.name, path)
            logging.info(f"Updated {path}")

    logging.info("All done – multi-pivot flags injected.")

if __name__ == "__main__":
    main()
