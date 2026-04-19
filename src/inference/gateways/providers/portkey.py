#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style MATH-500 inference via the Portkey AI Gateway (e.g., Princeton AI
Sandbox).

Behavior:
- Single-pass generation with the same math system prompt used in GRPO runs.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are
  generated.

This script is a **specialized / legacy gateway runner**. For canonical
HF-based math evaluation, prefer
``src.inference.cli.unified_math``; use this module only when you
need Portkey/sandbox-based remote models.

Auth:
- Expects an API key in the `AI_SANDBOX_KEY` env var (Portkey key or sandbox
  key).

Example usage:
  export AI_SANDBOX_KEY="***"
  python -m src.inference.gateways.providers.portkey \
      --output_dir artifacts/results/gpt4o-math-portkey
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict

from src.inference.domains.math.math_core import load_math500
from src.inference.gateways.base import setup_gateway_logger
from src.inference.utils.common import (
    GatewayCallParams,
    append_jsonl_row,
    build_second_pass_cue_strings,
    build_math_gateway_arg_parser,
    build_math_gateway_messages,
    build_math_gateway_row_base,
    build_usage_dict,
    iter_jsonl_objects,
    locked_file,
    resolve_output_dir_for_temperature,
)
from src.inference.utils.common import canon_math as _canon_math
from src.inference.utils.common import extract_blocks as _extract_blocks
from src.inference.utils.common import (
    extract_problem_and_answer,
    parse_openai_chat_response,
    prepare_math_gateway_dataset_from_args,
    require_datasets,
    setup_hf_cache_dir_env,
)
from src.inference.utils.common import valid_tag_structure as _valid_tag_structure
from src.inference.utils.task_registry import MATH_SYSTEM_PROMPT


DATASET_TYPE, load_dataset = require_datasets()
logger = setup_gateway_logger(__name__)


# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT
PASS2_KEYS = ("pass2a", "pass2b", "pass2c", "pass2d")


# ----------------------- Portkey client + call -----------------------
PortkeyCallParams = GatewayCallParams


@dataclass
class PortkeyRunConfig:
    """
    Configuration for a Portkey MATH-500 pass.

    :param output_path: Path to the JSONL file where results are written.
    :param split_name: Dataset split name (for example, ``\"test\"``).
    :param model_name: Model identifier used via Portkey.
    :param num_samples: Number of samples to generate per problem.
    :param params: Generation parameters such as temperature and limits.
    :param seed: Random seed for sampling and dataset shuffling.
    :param step: Training or checkpoint step identifier for filenames.
    :param two_pass: Whether to run second-pass cue interventions.
    :param second_pass_phrase: Raw cue string(s) for second-pass prompts.
    """

    output_path: str
    split_name: str
    model_name: str
    num_samples: int
    params: PortkeyCallParams
    seed: int
    step: int
    two_pass: bool = False
    second_pass_phrase: str = ""


@dataclass
class ExampleContext:
    """
    Per-example metadata for a MATH-500 row.

    :param problem: Normalized problem text.
    :param gold_answer: Ground-truth answer associated with the problem.
    :param canon_gold: Canonicalized gold answer.
    :param sample_idx: Sample index for this generation.
    """

    problem: str
    gold_answer: Any
    canon_gold: Any
    sample_idx: int


@dataclass
class PortkeyCallResult:
    """
    Result of a single Portkey generation call.

    :param text: Raw model output text.
    :param answer: Extracted answer text from the output.
    :param finish_reason: Finish reason reported by the API.
    :param usage: Optional usage object returned by the SDK.
    """

    text: str
    answer: str
    finish_reason: Any
    usage: Any


def _make_client():
    """
    Construct a Portkey client using ``AI_SANDBOX_KEY`` from the environment.

    :returns: Configured ``portkey_ai.Portkey`` client instance.
    :raises RuntimeError: If ``AI_SANDBOX_KEY`` is not set.
    :raises ImportError: If the ``portkey-ai`` package is not installed.
    """
    try:
        portkey_mod = import_module("portkey_ai")
    except ImportError as import_exc:  # pragma: no cover - optional dependency
        print(
            "portkey-ai is required for this script: pip install portkey-ai",
            file=sys.stderr,
        )
        raise import_exc

    api_key = os.getenv("AI_SANDBOX_KEY")
    if not api_key:
        raise RuntimeError("AI_SANDBOX_KEY env var is required for Portkey.")
    client_cls = getattr(portkey_mod, "Portkey")
    return client_cls(api_key=api_key)


def _call_model(
    client,
    model: str,
    problem: str,
    params: PortkeyCallParams,
):
    """
    Call the Portkey model for a single math problem.

    :param client: Portkey client created by :func:`_make_client`.
    :param model: Model identifier to use via Portkey.
    :param problem: Raw problem text to send to the model.
    :param params: Generation parameters such as temperature and limits.
    :returns: Parsed response tuple as returned by :func:`parse_openai_chat_response`.
    """
    messages = build_math_gateway_messages(SYSTEM_PROMPT, problem)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_output_tokens,
        timeout=params.request_timeout,
    )
    text, finish_reason, usage = parse_openai_chat_response(resp)
    if not text or not str(text).strip():
        raise RuntimeError(
            f"Empty model output (model={model}, finish_reason={finish_reason})"
        )
    return text, finish_reason, usage


def _call_model_with_messages(
    client,
    model: str,
    messages: list[dict[str, str]],
    params: PortkeyCallParams,
):
    """
    Call the Portkey model with an explicit chat message list.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_output_tokens,
        timeout=params.request_timeout,
    )
    text, finish_reason, usage = parse_openai_chat_response(resp)
    if not text or not str(text).strip():
        raise RuntimeError(
            f"Empty model output (model={model}, finish_reason={finish_reason})"
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


def _backfill_pass2(
    client,
    config: PortkeyRunConfig,
) -> None:
    tasks = [rec for rec in iter_jsonl_objects(config.output_path)]
    if not tasks:
        logger.info("No existing rows to backfill in %s", config.output_path)
        return

    cue_strs = build_second_pass_cue_strings(config.second_pass_phrase) if config.two_pass else []
    if not cue_strs:
        logger.info("No second-pass cues resolved; skipping backfill for %s", config.output_path)
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
            messages = _build_pass2_messages(record.get("problem", ""), prev_output.strip(), cue_text)
            text2, finish2, usage2 = _call_model_with_messages(
                client=client,
                model=config.model_name,
                messages=messages,
                params=config.params,
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
                config.output_path,
                problem=record.get("problem", ""),
                sample_idx=record.get("sample_idx", -1),
                updates=pass2_results,
            ):
                updated += 1
    logger.info("Backfill complete for %s | updated=%d", config.output_path, updated)


def _iter_examples(dataset: DATASET_TYPE, num_examples: int | None):
    """
    Yield at most ``num_examples`` examples from a dataset (or all if ``None``).

    :param dataset: Dataset object supporting ``select`` and iteration.
    :param num_examples: Maximum number of examples to yield, or ``None`` for all.
    :returns: Iterator over dataset examples.
    """
    if num_examples is not None and num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    yield from dataset


def _build_portkey_row(
    example: ExampleContext,
    result: PortkeyCallResult,
    config: PortkeyRunConfig,
) -> Dict[str, Any]:
    """
    Build a single JSONL row for Portkey MATH-500 inference and compute correctness.

    :param example: Per-example metadata describing problem and gold answer.
    :param result: Result from the Portkey model call.
    :param config: Run configuration including split, model, and step.
    :returns: Dictionary representing one JSONL row.
    """
    pred_canon = _canon_math(result.answer)
    is_correct = bool(pred_canon and example.canon_gold and example.canon_gold in pred_canon)

    row: Dict[str, Any] = build_math_gateway_row_base(
        problem=example.problem,
        gold_answer=example.gold_answer,
        gold_answer_canon=example.canon_gold,
        split=config.split_name,
        step=config.step,
        sample_idx=example.sample_idx,
    )
    row.update(
        {
            "endpoint": "portkey-ai",
            "deployment": config.model_name,
            "api_version": None,
            "temperature": config.params.temperature,
            "top_p": config.params.top_p,
            "pass1": {
                "output": result.text.strip(),
                "pred_answer": result.answer,
                "pred_answer_canon": pred_canon,
                "is_correct_pred": is_correct,
                "valid_tag_structure": _valid_tag_structure(result.text),
                "finish_reason": result.finish_reason,
            },
        },
    )

    if result.usage is not None:
        row["usage"] = build_usage_dict(result.usage)

    return row


def run_portkey_math_inference(
    client,
    dataset,
    existing: Dict[str, set],
    config: PortkeyRunConfig,
) -> None:
    """
    Run single-pass math inference via Portkey and write JSONL results.

    :param client: Portkey client created by :func:`_make_client`.
    :param dataset: Dataset object containing math problems and answers.
    :param existing: Mapping from problem to already-filled sample indices.
    :param config: Run configuration including output path and sampling options.
    :returns: ``None``. Results are appended to the JSONL file.
    """
    random.seed(config.seed)
    cue_strs = build_second_pass_cue_strings(config.second_pass_phrase) if config.two_pass else []
    total_cues = len(cue_strs)
    total_new = 0
    for example in _iter_examples(dataset, None):
        problem, gold_answer = extract_problem_and_answer(example)
        if not problem or gold_answer is None:
            continue

        todo_indices = [idx for idx in range(config.num_samples) if idx not in existing.get(problem, set())]
        if not todo_indices:
            continue

        for sample_idx in todo_indices:
            text, finish_reason, usage = _call_model(
                client=client,
                model=config.model_name,
                problem=problem,
                params=config.params,
            )

            _, answer = _extract_blocks(text)
            row = _build_portkey_row(
                ExampleContext(
                    problem=problem,
                    gold_answer=gold_answer,
                    canon_gold=_canon_math(gold_answer),
                    sample_idx=sample_idx,
                ),
                PortkeyCallResult(
                    text=text,
                    answer=answer,
                    finish_reason=finish_reason,
                    usage=usage,
                ),
                config,
            )

            if config.two_pass and cue_strs:
                pass2_results: Dict[str, Dict[str, Any]] = {}
                for idx, cue in enumerate(cue_strs):
                    if idx >= len(PASS2_KEYS):
                        break
                    is_neutral = total_cues >= 4 and idx == total_cues - 1
                    injected = not is_neutral
                    cue_text = cue.strip()
                    messages = _build_pass2_messages(problem, text.strip(), cue_text)
                    text2, finish2, usage2 = _call_model_with_messages(
                        client=client,
                        model=config.model_name,
                        messages=messages,
                        params=config.params,
                    )
                    pass2_results[PASS2_KEYS[idx]] = _build_pass_dict(
                        text=text2,
                        gold_answer_canon=_canon_math(gold_answer),
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

            append_jsonl_row(config.output_path, row)
            total_new += 1
            existing.setdefault(problem, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples → %s", total_new, config.output_path)


def main() -> None:
    """
    Parse arguments, load dataset, and run Portkey-based MATH-500 inference.

    :returns: ``None``. The function parses CLI args and runs the main loop.
    """
    parser = build_math_gateway_arg_parser(
        default_temperature=0.05,
        description="Portkey MATH-500 runner.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name to use via Portkey (e.g., gpt-4o, gpt-5, o3-mini).",
    )

    args = parser.parse_args()
    temps = getattr(args, "temperatures", None) or [args.temperature]
    multi = getattr(args, "temperatures", None) is not None

    client = _make_client()
    logger.info("Portkey client ready | model=%s", args.model)

    for temp in temps:
        random.seed(args.seed)
        temp_args = argparse.Namespace(**vars(args))
        temp_args.temperature = float(temp)
        output_dir = resolve_output_dir_for_temperature(
            args.output_dir,
            temp_args.temperature,
            multi=multi,
        )
        temp_args.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"step{temp_args.step:04d}_{temp_args.split}.jsonl",
        )
        call_params = PortkeyCallParams(
            temperature=temp_args.temperature,
            top_p=temp_args.top_p,
            max_output_tokens=temp_args.max_output_tokens,
            request_timeout=temp_args.request_timeout,
        )
        config = PortkeyRunConfig(
            output_path=output_path,
            split_name=temp_args.split,
            model_name=temp_args.model,
            num_samples=temp_args.num_samples,
            params=call_params,
            seed=temp_args.seed,
            step=temp_args.step,
            two_pass=bool(getattr(temp_args, "two_pass", False)),
            second_pass_phrase=str(getattr(temp_args, "second_pass_phrase", "") or ""),
        )
        if config.two_pass and getattr(temp_args, "backfill_pass2", False):
            _backfill_pass2(client=client, config=config)
            continue

        dataset, existing, _ = prepare_math_gateway_dataset_from_args(
            args=temp_args,
            outpath=output_path,
            logger=logger,
            load_math500_fn=load_math500,
            load_remote_dataset_fn=load_dataset,
            cache_dir=setup_hf_cache_dir_env("./.hf_cache"),
        )
        logger.info("Running temperature=%s -> %s", temp_args.temperature, output_path)
        run_portkey_math_inference(
            client=client,
            dataset=dataset,
            existing=existing,
            config=config,
        )


if __name__ == "__main__":
    main()
