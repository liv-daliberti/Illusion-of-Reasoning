#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style MATH-500 inference for DeepSeek-R1 via OpenRouter.

Behavior:
- Single-pass generation with the same math system prompt used in GRPO runs.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are generated.

This script is a **specialized / legacy gateway runner** intended for remote
API experiments. For canonical HF-based math evaluation, prefer
``src.inference.cli.unified_math``.

Auth:
- Expects an OpenRouter API key in the `OPENROUTER_API_KEY` env var.

Example usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python -m src.inference.gateways.providers.openrouter \\
    --output_dir artifacts/results/deepseek-r1-openrouter \\
    --model deepseek/deepseek-r1:free
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sys
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any, Dict

from src.inference.domains.math.math_core import load_math500
from src.inference.gateways.base import setup_gateway_logger
from src.inference.utils import gateway_utils as _gateway_utils
from src.inference.utils.math_pass_utils import build_second_pass_cue_strings
from src.inference.utils.math_pass_utils import canon_math as _canon_math
from src.inference.utils.math_pass_utils import extract_blocks as _extract_blocks
from src.inference.utils.math_pass_utils import valid_tag_structure as _valid_tag_structure
from src.inference.utils.task_registry import MATH_SYSTEM_PROMPT


DATASET_TYPE, load_dataset = _gateway_utils.require_datasets()
logger = setup_gateway_logger(__name__)

# Bind gateway helpers with a defensive RetryContext fallback so stubbed tests
# lacking the attribute do not fail at import time.
append_jsonl_row = _gateway_utils.append_jsonl_row
build_math_gateway_arg_parser = _gateway_utils.build_math_gateway_arg_parser
build_math_gateway_messages = _gateway_utils.build_math_gateway_messages
build_math_gateway_row_base = _gateway_utils.build_math_gateway_row_base
build_usage_dict = _gateway_utils.build_usage_dict
locked_file = _gateway_utils.locked_file
call_with_gateway_retries = _gateway_utils.call_with_gateway_retries
call_with_gateway_retries_compat = getattr(
    _gateway_utils,
    "call_with_gateway_retries_compat",
    None,
)
RetryContext = getattr(_gateway_utils, "RetryContext", None)
if RetryContext is None:  # pragma: no cover - exercised in stubbed tests

    @dataclass(frozen=True)
    class RetryContext:  # type: ignore[redefinition]
        """Stub retry context for environments lacking gateway RetryContext."""

        logger: Any
        sample_idx: int
        problem_snippet: str
        min_sleep: float | None = None


iter_jsonl_objects = _gateway_utils.iter_jsonl_objects
iter_math_gateway_samples = _gateway_utils.iter_math_gateway_samples
parse_openai_chat_response = _gateway_utils.parse_openai_chat_response
prepare_math_gateway_dataset_from_args = _gateway_utils.prepare_math_gateway_dataset_from_args


# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT

PASS2_KEYS = ("pass2a", "pass2b", "pass2c", "pass2d")


# ----------------------- OpenRouter client + call -----------------------
def _make_client():
    """
    Construct an OpenAI client pointed at the OpenRouter base URL.

    :returns: Configured ``openai.OpenAI`` client instance for OpenRouter.
    :raises RuntimeError: If ``OPENROUTER_API_KEY`` is not set in the environment.
    :raises ImportError: If the ``openai`` package is not installed.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY env var is required for OpenRouter.")
    base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    try:
        openai_mod = import_module("openai")
    except ImportError as import_exc:  # pragma: no cover - optional dependency
        print(
            "openai>=1.x is required for this script: pip install openai",
            file=sys.stderr,
        )
        raise import_exc

    client_cls = getattr(openai_mod, "OpenAI")
    return client_cls(base_url=base_url, api_key=api_key)


def _call_model(client, problem: str, args: argparse.Namespace):
    """
    Call the OpenRouter model for a single math problem.

    :param client: OpenRouter client returned by :func:`_make_client`.
    :param problem: Raw problem text to send to the model.
    :param args: Parsed CLI arguments containing model and sampling options.
    :returns: Parsed response tuple as returned by :func:`parse_openai_chat_response`.
    """
    # If you want OpenRouter's explicit reasoning traces, uncomment extra_body.
    messages = build_math_gateway_messages(SYSTEM_PROMPT, problem)
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        timeout=args.request_timeout,
        # extra_body={"reasoning": True},
    )
    text, finish_reason, usage = parse_openai_chat_response(resp)
    if not text or not str(text).strip():
        raise RuntimeError(
            f"Empty model output (model={args.model}, finish_reason={finish_reason})"
        )
    return text, finish_reason, usage


def _call_model_with_messages(client, messages: list[dict[str, str]], args: argparse.Namespace):
    """
    Call the OpenRouter model with an explicit chat message list.
    """
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        timeout=args.request_timeout,
        # extra_body={"reasoning": True},
    )
    text, finish_reason, usage = parse_openai_chat_response(resp)
    if not text or not str(text).strip():
        raise RuntimeError(
            f"Empty model output (model={args.model}, finish_reason={finish_reason})"
        )
    return text, finish_reason, usage


def _build_pass2_messages(problem: str, prev_output: str, cue: str) -> list[dict[str, str]]:
    """
    Build a chat history for second-pass generation with an injected cue.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": prev_output},
        {"role": "user", "content": cue},
    ]


def _build_pass_dict(
    *,
    text: str,
    gold_answer_canon: str | None,
    finish_reason: str | None,
    usage: Any,
    cue_text: str | None = None,
    injected_cue: bool = False,
) -> Dict[str, Any]:
    """
    Build a pass dictionary for pass2 variants (or pass1 when cue_text is None).
    """
    _, ans = _extract_blocks(text)
    pred_canon = _canon_math(ans)
    is_correct = bool(pred_canon and gold_answer_canon and gold_answer_canon in pred_canon)
    pass_dict: Dict[str, Any] = {
        "output": text.strip(),
        "pred_answer": ans,
        "pred_answer_canon": pred_canon,
        "is_correct_pred": is_correct,
        "valid_tag_structure": _valid_tag_structure(text),
        "finish_reason": finish_reason,
    }
    if cue_text is not None:
        pass_dict["cue_text"] = cue_text
        pass_dict["has_reconsider_cue"] = bool(injected_cue)
        pass_dict["reconsider_markers"] = ["injected_cue"] if injected_cue else []
        pass_dict["is_correct_after_reconsideration"] = bool(injected_cue) and bool(is_correct)
    if usage is not None:
        pass_dict["usage"] = build_usage_dict(usage)
    return pass_dict


def _pass_missing(record: Dict[str, Any], key: str) -> bool:
    value = record.get(key)
    return value is None or value == {}


def _write_jsonl_records(path: str, records: list[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
    os.replace(tmp_path, path)


def _update_record_with_lock(
    path: str,
    *,
    problem: str,
    sample_idx: int,
    updates: Dict[str, Dict[str, Any]],
) -> bool:
    if not os.path.exists(path):
        return False
    updated = False
    with locked_file(path, "r+", lock_type=fcntl.LOCK_EX) as handle:
        handle.seek(0)
        records: list[Dict[str, Any]] = []
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("problem") == problem
                and int(record.get("sample_idx", -1)) == int(sample_idx)
            ):
                for key, value in updates.items():
                    if _pass_missing(record, key):
                        record[key] = value
                        updated = True
                if _pass_missing(record, "pass2"):
                    if record.get("pass2c"):
                        record["pass2"] = record["pass2c"]
                        updated = True
                    elif updates:
                        last_key = max(updates, key=lambda k: PASS2_KEYS.index(k))
                        record["pass2"] = record[last_key]
                        updated = True
            records.append(record)
        if updated:
            handle.seek(0)
            handle.truncate()
            for record in records:
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
    return updated


def _backfill_pass2(client, args: argparse.Namespace, outpath: str) -> None:
    tasks = [rec for rec in iter_jsonl_objects(outpath)]
    if not tasks:
        logger.info("No existing rows to backfill in %s", outpath)
        return

    cue_strs = build_second_pass_cue_strings(getattr(args, "second_pass_phrase", "") or "")
    if not cue_strs:
        logger.info("No second-pass cues resolved; skipping backfill for %s", outpath)
        return

    total_cues = len(cue_strs)
    updated = 0
    for record in tasks:
        pass1 = record.get("pass1") or {}
        prev_output = pass1.get("output")
        if not isinstance(prev_output, str) or not prev_output.strip():
            continue

        gold_answer_canon = record.get("gold_answer_canon")
        pass2_results: Dict[str, Dict[str, Any]] = {}
        for idx, cue in enumerate(cue_strs):
            if idx >= len(PASS2_KEYS):
                break
            key = PASS2_KEYS[idx]
            if not _pass_missing(record, key):
                continue
            is_neutral = total_cues >= 4 and idx == total_cues - 1
            injected = not is_neutral
            cue_text = cue.strip()
            retry_ctx = _gateway_utils.build_retry_context(
                logger=logger,
                sample_idx=record.get("sample_idx", -1),
                problem_snippet=f"{record.get('problem', '')} | cue{idx + 1}",
            )
            messages = _build_pass2_messages(record.get("problem", ""), prev_output.strip(), cue_text)
            text2, finish2, usage2 = call_with_gateway_retries_compat(
                call_with_gateway_retries,
                partial(_call_model_with_messages, client, messages, args),
                args,
                retry_ctx,
            )
            pass2_results[key] = _build_pass_dict(
                text=text2,
                gold_answer_canon=gold_answer_canon,
                finish_reason=finish2,
                usage=usage2,
                cue_text=cue_text,
                injected_cue=injected,
            )
        if pass2_results:
            if _update_record_with_lock(
                outpath,
                problem=record.get("problem", ""),
                sample_idx=record.get("sample_idx", -1),
                updates=pass2_results,
            ):
                updated += 1
    logger.info("Backfill complete for %s | updated=%d", outpath, updated)


# ----------------------- Main helpers + loop -----------------------
def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the OpenRouter DeepSeek-R1 runner.

    :returns: Parsed :class:`argparse.Namespace` with configuration values.
    """
    parser = build_math_gateway_arg_parser(
        default_temperature=0.05,
        description="OpenRouter DeepSeek-R1 MATH-500 runner.",
    )
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-r1",
        help="OpenRouter model name (e.g., deepseek/deepseek-r1 or deepseek/deepseek-r1:free).",
    )
    parser.add_argument("--max_retries", type=int, default=15)
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=10.0,
        help="Base backoff in seconds; actual sleep is max(10, retry_backoff * attempt).",
    )
    return parser.parse_args()


def _prepare_dataset(
    args: argparse.Namespace,
    outpath: str,
) -> tuple[DATASET_TYPE, Dict[str, set[int]], str]:
    """
    Load, optionally subsample, and shuffle the dataset.

    :param args: Parsed CLI arguments controlling dataset choice and sampling.
    :param outpath: Output path used to determine resume/fill behavior.
    :returns: Tuple ``(dataset, existing, dataset_name_for_log)`` where
        ``existing`` maps problems to the set of already-filled sample indices.
    """
    dataset, existing, dataset_name_for_log = prepare_math_gateway_dataset_from_args(
        args=args,
        outpath=outpath,
        logger=logger,
        load_math500_fn=load_math500,
        load_remote_dataset_fn=load_dataset,
    )
    return dataset, existing, dataset_name_for_log


def _generate_samples(client, args: argparse.Namespace, outpath: str) -> None:
    """
    Main generation loop over the dataset.

    :param client: OpenRouter client used to issue chat completions.
    :param args: Parsed CLI arguments controlling generation behavior.
    :param outpath: Path to the JSONL file where results are written.
    :returns: ``None``. Samples are written to disk and progress is logged.
    """
    dataset, existing, _ = _prepare_dataset(args, outpath)

    total_new = 0
    for problem, gold_answer, sample_idx in iter_math_gateway_samples(
        dataset,
        args.num_samples,
        existing,
    ):
        retry_ctx = _gateway_utils.build_retry_context(
            logger=logger,
            sample_idx=sample_idx,
            problem_snippet=problem,
        )
        text, finish_reason, usage = call_with_gateway_retries_compat(
            call_with_gateway_retries,
            partial(_call_model, client, problem, args),
            args,
            retry_ctx,
        )

        _, ans = _extract_blocks(text)

        row: Dict[str, Any] = build_math_gateway_row_base(
            problem=problem,
            gold_answer=gold_answer,
            gold_answer_canon=_canon_math(gold_answer),
            split=args.split,
            step=args.step,
            sample_idx=sample_idx,
        )
        row.update(
            {
                "endpoint": "openrouter",
                "deployment": args.model,
                "api_version": None,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "pass1": {
                    "output": text.strip(),
                    "pred_answer": ans,
                    "pred_answer_canon": _canon_math(ans),
                    "is_correct_pred": bool(
                        _canon_math(ans) and row["gold_answer_canon"] and row["gold_answer_canon"] in _canon_math(ans)
                    ),
                    "valid_tag_structure": _valid_tag_structure(text),
                    "finish_reason": finish_reason,
                },
            },
        )

        if usage is not None:
            row["usage"] = build_usage_dict(usage)

        if getattr(args, "two_pass", False):
            cue_strs = build_second_pass_cue_strings(getattr(args, "second_pass_phrase", "") or "")
            total_cues = len(cue_strs)
            pass2_results: Dict[str, Dict[str, Any]] = {}
            for idx, cue in enumerate(cue_strs):
                if idx >= len(PASS2_KEYS):
                    break
                is_neutral = total_cues >= 4 and idx == total_cues - 1
                injected = not is_neutral
                cue_text = cue.strip()
                retry_ctx = _gateway_utils.build_retry_context(
                    logger=logger,
                    sample_idx=sample_idx,
                    problem_snippet=f"{problem} | cue{idx + 1}",
                )
                messages = _build_pass2_messages(problem, text.strip(), cue_text)
                text2, finish2, usage2 = call_with_gateway_retries_compat(
                    call_with_gateway_retries,
                    partial(_call_model_with_messages, client, messages, args),
                    args,
                    retry_ctx,
                )
                pass2_results[PASS2_KEYS[idx]] = _build_pass_dict(
                    text=text2,
                    gold_answer_canon=row["gold_answer_canon"],
                    finish_reason=finish2,
                    usage=usage2,
                    cue_text=cue_text,
                    injected_cue=injected,
                )
            for key, pass_dict in pass2_results.items():
                row[key] = pass_dict
            if "pass2c" in pass2_results:
                row["pass2"] = pass2_results["pass2c"]
            elif pass2_results:
                row["pass2"] = pass2_results[PASS2_KEYS[len(pass2_results) - 1]]

        append_jsonl_row(outpath, row)
        total_new += 1
        existing.setdefault(problem, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples → %s", total_new, outpath)


def main() -> None:
    """
    CLI entry point for generating DeepSeek-R1 samples via OpenRouter.

    :returns: ``None``. The function parses arguments and runs the generation loop.
    """
    args = _parse_args()
    temps = getattr(args, "temperatures", None) or [args.temperature]
    multi = getattr(args, "temperatures", None) is not None

    client = _make_client()
    logger.info("OpenRouter client ready | model=%s", args.model)

    for temp in temps:
        random.seed(args.seed)
        temp_args = argparse.Namespace(**vars(args))
        temp_args.temperature = float(temp)
        output_dir = _gateway_utils.resolve_output_dir_for_temperature(
            args.output_dir,
            temp_args.temperature,
            multi=multi,
        )
        temp_args.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"step{temp_args.step:04d}_{temp_args.split}.jsonl")
        logger.info("Running temperature=%s -> %s", temp_args.temperature, outpath)
        if getattr(temp_args, "backfill_pass2", False) and getattr(temp_args, "two_pass", False):
            _backfill_pass2(client, temp_args, outpath)
            continue
        _generate_samples(client, temp_args, outpath)


if __name__ == "__main__":
    main()
