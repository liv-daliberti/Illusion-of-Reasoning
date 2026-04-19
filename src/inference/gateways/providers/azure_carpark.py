#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style Rush Hour (car-park) inference for Azure-hosted models.

Behavior:
- Single-pass generation with optional two-pass reconsideration cues.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are
  generated.
- Uses Azure OpenAI (Responses API if available; falls back to Chat
  Completions).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

from src.annotate import build_preferred_client, load_azure_config
from src.inference.domains.carpark.carpark_board import (
    _canon_rush_generic,
    _canon_rush_gold,
    _is_valid_rush,
)
from src.inference.domains.carpark.carpark_data import (
    SYSTEM_PROMPT,
    load_existing_example_index,
    load_rush_dataset,
    norm_fields,
)
from src.inference.gateways.base import setup_gateway_logger
from src.inference.utils import gateway_utils as _gateway_utils
from src.inference.utils.carpark_rush_utils import rush_soft_match_reward
from src.inference.utils.math_pass_utils import (
    build_second_pass_cue_strings,
    extract_blocks as _extract_blocks,
    valid_tag_structure as _valid_tag_structure,
)


logger = setup_gateway_logger(__name__)

# RuntimeError("Empty model output ...") can still bubble after retries.
_EMPTY_OUTPUT_MARKER = "Empty model output"

# ----------------------- Prompt -----------------------
PASS2_KEYS = ("pass2a", "pass2b", "pass2c", "pass2d")

# Optional reconsideration markers (analytics); kept lightweight to avoid
# importing the full carpark solver module.
_RECONSIDER_PATTERNS = [
    ("wait_line", re.compile(r"(?im)^\s*wait[,\.\- ]", re.I)),
    ("wait_reconsider", re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("step_by_step", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck", re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]

# Bind common helpers with defensive fallbacks for stubbed environments.
append_jsonl_row = _gateway_utils.append_jsonl_row
build_math_gateway_arg_parser = _gateway_utils.build_math_gateway_arg_parser
build_retry_context = _gateway_utils.build_retry_context
build_two_pass_row_base = _gateway_utils.build_two_pass_row_base
build_usage_dict = _gateway_utils.build_usage_dict
call_with_gateway_retries = _gateway_utils.call_with_gateway_retries
call_with_gateway_retries_compat = _gateway_utils.call_with_gateway_retries_compat
iter_jsonl_objects = _gateway_utils.iter_jsonl_objects
limit_dataset_examples = _gateway_utils.limit_dataset_examples
load_local_json_dataset = _gateway_utils.load_local_json_dataset
locked_file = _gateway_utils.locked_file
resolve_output_dir_for_temperature = _gateway_utils.resolve_output_dir_for_temperature
setup_hf_cache_dir_env = _gateway_utils.setup_hf_cache_dir_env

GatewayCallParams = _gateway_utils.GatewayCallParams
AzureCallParams = GatewayCallParams


def _ensure_system_message(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if any(msg.get("role") == "system" for msg in messages):
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)


def _messages_to_problem_text(messages: list[dict[str, str]]) -> str:
    return " ".join(str(msg.get("content", "")) for msg in messages)


def _find_reconsider_markers(think_text: str, *, injected_cue: bool) -> list[str]:
    markers: list[str] = []
    if think_text:
        for name, pattern in _RECONSIDER_PATTERNS:
            if pattern.search(think_text):
                markers.append(name)
                break
    if injected_cue:
        markers = ["injected_cue"] + markers
    return markers


def _prepare_dataset(args: argparse.Namespace, outpath: str):
    cache_dir = setup_hf_cache_dir_env("./.hf_cache")
    if args.dataset_path:
        dataset = load_local_json_dataset(args.dataset_path)
        dataset_name = os.path.basename(args.dataset_path)
    else:
        dataset = load_rush_dataset(
            args.dataset_id,
            split=args.split,
            cache_dir=cache_dir,
            prompt_col=args.dataset_prompt_column,
            solution_col=args.dataset_solution_column,
        )
        dataset_name = args.dataset_id

    columns = set(dataset.column_names)
    if args.dataset_prompt_column not in columns or args.dataset_solution_column not in columns:
        raise ValueError(
            "Dataset missing required columns: "
            f"{args.dataset_prompt_column}, {args.dataset_solution_column}. "
            f"Found: {sorted(columns)}"
        )
    if "id" not in columns:
        dataset = dataset.map(lambda _example, idx: {"id": f"idx_{idx}"}, with_indices=True)

    dataset = limit_dataset_examples(
        dataset,
        args.num_examples,
        from_end=getattr(args, "examples_from_end", False),
        start=getattr(args, "dataset_start", 0),
    )
    dataset = dataset.shuffle(seed=args.seed)
    existing = load_existing_example_index(outpath)
    logger.info(
        "Dataset: %s split=%s | N=%d | existing=%d",
        dataset_name,
        args.split,
        len(dataset),
        len(existing),
    )
    logger.info("Output: %s", outpath)
    return dataset, existing


def _iter_rush_samples(
    dataset,
    num_samples: int,
    existing: Dict[str, set[int]],
    prompt_col: str,
    solution_col: str,
) -> Iterable[Tuple[str, list[dict[str, str]], Any, int]]:
    for example in dataset:
        messages, solution = norm_fields(example, prompt_col, solution_col)
        example_id = str(example.get("id"))
        generated_indices = existing.get(example_id, set())
        for sample_idx in range(num_samples):
            if sample_idx in generated_indices:
                continue
            yield example_id, messages, solution, sample_idx


def _call_with_retries_compat(func, args: argparse.Namespace, sample_idx: int, problem: str):
    retry_ctx = build_retry_context(
        logger=logger,
        sample_idx=sample_idx,
        problem_snippet=problem,
    )
    try:
        return call_with_gateway_retries_compat(
            call_with_gateway_retries,
            func,
            args,
            retry_ctx,
        )
    except TypeError:
        return call_with_gateway_retries(func, args=args, context=retry_ctx)


@dataclass
class AzureResultRowInput:
    """Bundled inputs needed to construct a JSONL result row."""

    example_id: str
    messages: list[dict[str, str]]
    gold_answer: Any
    sample_idx: int
    text: str
    finish_reason: Any
    usage: Any


def _make_client(args: argparse.Namespace):
    cfg = load_azure_config()
    endpoint = (args.endpoint or cfg["endpoint"]).rstrip("/")
    deployment = args.deployment or cfg["deployment"]
    api_version = args.api_version or cfg["api_version"]
    api_key = args.api_key or os.getenv(
        "AZURE_OPENAI_API_KEY",
        cfg.get("api_key", ""),
    )
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is required (env or --api_key).")
    client, uses_v1 = build_preferred_client(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        use_v1=bool(args.use_v1),
    )
    return client, uses_v1, endpoint, deployment, api_version


def _call_model_with_messages(
    client,
    uses_v1: bool,
    deployment: str,
    messages: list[dict[str, str]],
    params: AzureCallParams,
):
    if uses_v1 and hasattr(client, "responses"):
        has_system = any(msg.get("role") == "system" for msg in messages)
        input_messages = messages if has_system else [msg for msg in messages if msg.get("role") != "system"]
        request = {
            "model": deployment,
            "input": input_messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_output_tokens": params.max_output_tokens,
            "timeout": params.request_timeout,
        }
        if not has_system:
            request["instructions"] = SYSTEM_PROMPT
        resp = client.responses.create(**request)
        text = ""
        finish_reason = None
        if getattr(resp, "output", None):
            output = resp.output
            if getattr(output, "choices", None):
                choice = output.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                msg = getattr(choice, "message", None)
                text = getattr(msg, "content", "") if msg is not None else ""
        usage = getattr(resp, "usage", None)
        if not text or not str(text).strip():
            raise RuntimeError(
                f"Empty model output (deployment={deployment}, finish_reason={finish_reason})"
            )
        return text, finish_reason, usage

    resp = client.chat.completions.create(
        model=deployment,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_output_tokens,
        timeout=params.request_timeout,
        messages=messages,
    )
    finish_reason = None
    if resp and getattr(resp, "choices", None):
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        text = resp.choices[0].message.content or ""
    else:
        text = ""
    usage = getattr(resp, "usage", None)
    if not text or not str(text).strip():
        raise RuntimeError(
            f"Empty model output (deployment={deployment}, finish_reason={finish_reason})"
        )
    return text, finish_reason, usage


def _is_empty_output_error(exc: Exception) -> bool:
    """Return True if an exception indicates an empty model output."""
    return _EMPTY_OUTPUT_MARKER in str(exc)


def _build_pass2_messages(
    messages: list[dict[str, str]],
    prev_output: str,
    cue: str,
) -> list[dict[str, str]]:
    return list(messages) + [
        {"role": "assistant", "content": prev_output},
        {"role": "user", "content": cue},
    ]


def _build_pass_dict(
    *,
    text: str,
    gold_answer: Any,
    gold_set: set[str],
    finish_reason: Any,
    usage: Any,
    prev_output: Optional[str] = None,
    cue_text: Optional[str] = None,
    injected_cue: bool = False,
) -> Dict[str, Any]:
    think_text, ans = _extract_blocks(text)
    pred_canon = _canon_rush_generic(ans)
    is_correct = bool(pred_canon and pred_canon in gold_set)
    is_valid = _is_valid_rush(pred_canon)
    soft_reward, soft_detail = rush_soft_match_reward(ans or "", gold_answer)
    markers = _find_reconsider_markers(think_text or "", injected_cue=injected_cue)

    pass_dict: Dict[str, Any] = {
        "output": text.strip(),
        "pred_answer": ans,
        "pred_answer_canon": pred_canon,
        "is_valid_pred": is_valid,
        "is_correct_pred": is_correct,
        "soft_reward": soft_reward,
        "soft_reward_detail": soft_detail,
        "valid_tag_structure": _valid_tag_structure(text),
        "finish_reason": finish_reason,
        "has_reconsider_cue": bool(markers),
        "reconsider_markers": markers,
    }
    if prev_output is not None:
        pass_dict["prev_output"] = prev_output
    if cue_text is not None:
        pass_dict["cue_text"] = cue_text
        pass_dict["is_correct_after_reconsideration"] = bool(injected_cue) and bool(is_correct)
    if usage is not None:
        pass_dict["usage"] = build_usage_dict(usage)
    return pass_dict


def _pass_missing(record: Dict[str, Any], key: str) -> bool:
    value = record.get(key)
    return value is None or value == {}


def _update_record_with_lock(
    path: str,
    *,
    example_id: str,
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
                str(record.get("example_id")) == str(example_id)
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
    uses_v1: bool,
    args: argparse.Namespace,
    call_params: AzureCallParams,
    output_path: str,
) -> None:
    tasks = [rec for rec in iter_jsonl_objects(output_path)]
    if not tasks:
        logger.info("No existing rows to backfill in %s", output_path)
        return

    cue_strs = build_second_pass_cue_strings(getattr(args, "second_pass_phrase", "") or "")
    if not cue_strs:
        logger.info("No second-pass cues resolved; skipping backfill for %s", output_path)
        return

    total_cues = len(cue_strs)
    updated = 0
    for record in tasks:
        pass1 = record.get("pass1") or {}
        prev_output = pass1.get("output")
        if not isinstance(prev_output, str) or not prev_output.strip():
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.get("problem", "")},
        ]
        gold_answer = record.get("gold_answer")
        gold_set = _canon_rush_gold(gold_answer)
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
            messages2 = _build_pass2_messages(messages, prev_output.strip(), cue_text)
            call_fn = partial(
                _call_model_with_messages,
                client,
                uses_v1,
                args.deployment,
                messages2,
                call_params,
            )
            try:
                call_result2 = _call_with_retries_compat(
                    call_fn,
                    args,
                    record.get("sample_idx", -1),
                    f"{record.get('example_id', '')} | cue{idx + 1}",
                )
            except RuntimeError as exc:
                if _is_empty_output_error(exc):
                    logger.warning(
                        "Skipping empty pass2 output after retries (id=%s cue=%s).",
                        record.get("example_id", -1),
                        key,
                    )
                    continue
                raise
            pass2_result = _build_pass_dict(
                text=call_result2[0],
                gold_answer=gold_answer,
                gold_set=gold_set,
                finish_reason=call_result2[1],
                usage=call_result2[2],
                prev_output=prev_output.strip(),
                cue_text=cue_text,
                injected_cue=injected,
            )
            pass2_result["improved_over_pass1"] = bool(pass2_result["is_correct_pred"]) and not bool(
                pass1.get("is_correct_pred")
            )
            pass2_results[key] = pass2_result
        if pass2_results:
            if _update_record_with_lock(
                output_path,
                example_id=record.get("example_id", ""),
                sample_idx=record.get("sample_idx", -1),
                updates=pass2_results,
            ):
                updated += 1
    logger.info("Backfill complete for %s | updated=%d", output_path, updated)


def _generate_samples(
    client,
    uses_v1: bool,
    args: argparse.Namespace,
    call_params: AzureCallParams,
    output_path: str,
) -> None:
    dataset, existing = _prepare_dataset(args, output_path)

    total_new = 0
    sample_iter = _iter_rush_samples(
        dataset,
        args.num_samples,
        existing,
        args.dataset_prompt_column,
        args.dataset_solution_column,
    )
    for example_id, messages, gold_answer, sample_idx in sample_iter:
        messages = _ensure_system_message(messages)
        call_fn = partial(
            _call_model_with_messages,
            client,
            uses_v1,
            args.deployment,
            messages,
            call_params,
        )
        try:
            call_result = _call_with_retries_compat(
                call_fn,
                args,
                sample_idx,
                example_id,
            )
        except RuntimeError as exc:
            if _is_empty_output_error(exc):
                logger.warning(
                    "Skipping empty pass1 output after retries (sample_idx=%s example_id=%s).",
                    sample_idx,
                    example_id,
                )
                continue
            raise

        row = _build_result_row(
            example_id=example_id,
            messages=messages,
            gold_answer=gold_answer,
            sample_idx=sample_idx,
            text=call_result[0],
            finish_reason=call_result[1],
            usage=call_result[2],
            args=args,
            call_params=call_params,
        )

        if getattr(args, "two_pass", False):
            cue_strs = build_second_pass_cue_strings(getattr(args, "second_pass_phrase", "") or "")
            total_cues = len(cue_strs)
            pass2_results: Dict[str, Dict[str, Any]] = {}
            pass1_output = row["pass1"]["output"].strip()
            for idx, cue in enumerate(cue_strs):
                if idx >= len(PASS2_KEYS):
                    break
                is_neutral = total_cues >= 4 and idx == total_cues - 1
                injected = not is_neutral
                cue_text = cue.strip()
                messages2 = _build_pass2_messages(messages, pass1_output, cue_text)
                call_fn = partial(
                    _call_model_with_messages,
                    client,
                    uses_v1,
                    args.deployment,
                    messages2,
                    call_params,
                )
                try:
                    call_result2 = _call_with_retries_compat(
                        call_fn,
                        args,
                        sample_idx,
                        f"{example_id} | cue{idx + 1}",
                    )
                except RuntimeError as exc:
                    if _is_empty_output_error(exc):
                        logger.warning(
                            "Skipping empty pass2 output after retries (sample_idx=%s cue=%s).",
                            sample_idx,
                            PASS2_KEYS[idx],
                        )
                        continue
                    raise
                pass2_result = _build_pass_dict(
                    text=call_result2[0],
                    gold_answer=gold_answer,
                    gold_set=set(row["gold_answer_canon_set"] or []),
                    finish_reason=call_result2[1],
                    usage=call_result2[2],
                    prev_output=pass1_output,
                    cue_text=cue_text,
                    injected_cue=injected,
                )
                pass2_result["improved_over_pass1"] = bool(pass2_result["is_correct_pred"]) and not bool(
                    row["pass1"]["is_correct_pred"]
                )
                pass2_results[PASS2_KEYS[idx]] = pass2_result

            for key, pass_dict in pass2_results.items():
                row[key] = pass_dict
            if "pass2c" in pass2_results:
                row["pass2"] = pass2_results["pass2c"]
            elif pass2_results:
                row["pass2"] = pass2_results[PASS2_KEYS[len(pass2_results) - 1]]

        append_jsonl_row(output_path, row)
        total_new += 1
        existing.setdefault(example_id, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples -> %s", total_new, output_path)


def _build_result_row(
    result: AzureResultRowInput | None = None,
    args: argparse.Namespace | None = None,
    call_params: AzureCallParams | None = None,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    if result is None:
        required_keys = ("example_id", "messages", "gold_answer", "sample_idx", "text")
        missing = [key for key in required_keys if key not in legacy_kwargs]
        if missing:
            raise TypeError(f"_build_result_row() missing required arguments: {', '.join(missing)}")
        result = AzureResultRowInput(
            example_id=legacy_kwargs["example_id"],
            messages=legacy_kwargs["messages"],
            gold_answer=legacy_kwargs["gold_answer"],
            sample_idx=legacy_kwargs["sample_idx"],
            text=legacy_kwargs["text"],
            finish_reason=legacy_kwargs.get("finish_reason"),
            usage=legacy_kwargs.get("usage"),
        )
    if args is None or call_params is None:
        raise TypeError("_build_result_row() requires args and call_params")

    gold_set = _canon_rush_gold(result.gold_answer)
    pass1 = _build_pass_dict(
        text=result.text,
        gold_answer=result.gold_answer,
        gold_set=gold_set,
        finish_reason=result.finish_reason,
        usage=None,
        prev_output=None,
    )

    row: Dict[str, Any] = {
        "example_id": result.example_id,
        "problem": _messages_to_problem_text(result.messages),
        "gold_answer": result.gold_answer,
        "gold_answer_canon_set": sorted(list(gold_set)),
        "endpoint": args.endpoint,
        "deployment": args.deployment,
        "api_version": args.api_version,
        "temperature": call_params.temperature,
        "top_p": call_params.top_p,
        **build_two_pass_row_base(
            step=args.step,
            split_name=args.split,
            sample_idx=result.sample_idx,
            pass1=pass1,
            pass2=None,
        ),
    }

    if result.usage:
        row["usage"] = build_usage_dict(result.usage)

    return row


def _parse_args() -> argparse.Namespace:
    default_cfg = load_azure_config()
    parser = build_math_gateway_arg_parser(
        default_temperature=0.7,
        description="Azure Rush Hour (car-park) gateway runner.",
    )
    parser.set_defaults(dataset_id="od2961/rush4-5-6-balanced")

    parser.add_argument("--dataset_prompt_column", default="messages")
    parser.add_argument("--dataset_solution_column", default="solution")

    parser.add_argument("--endpoint", default=default_cfg.get("endpoint"))
    parser.add_argument(
        "--deployment",
        default=None,
        help="Azure deployment name (e.g., gpt-4o).",
    )
    parser.add_argument("--api_version", default=default_cfg.get("api_version"))
    parser.add_argument("--api_key", default=None)
    parser.add_argument(
        "--use_v1",
        type=int,
        default=int(default_cfg.get("use_v1", 1)),
        help="1 -> prefer Responses API (v1); 0 -> force Chat Completions.",
    )

    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff", type=float, default=2.0)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    temps = getattr(args, "temperatures", None) or [args.temperature]
    multi = getattr(args, "temperatures", None) is not None
    client, uses_v1, endpoint, deployment, api_version = _make_client(args)
    logger.info(
        "Client ready | uses_v1=%s | endpoint=%s | deployment=%s",
        uses_v1,
        endpoint,
        deployment,
    )

    args.endpoint = endpoint
    args.deployment = deployment
    args.api_version = api_version

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
        call_params = AzureCallParams(
            temperature=temp_args.temperature,
            top_p=temp_args.top_p,
            max_output_tokens=temp_args.max_output_tokens,
            request_timeout=temp_args.request_timeout,
        )
        logger.info("Running temperature=%s -> %s", temp_args.temperature, output_path)
        if getattr(temp_args, "backfill_pass2", False) and getattr(temp_args, "two_pass", False):
            _backfill_pass2(
                client=client,
                uses_v1=uses_v1,
                args=temp_args,
                call_params=call_params,
                output_path=output_path,
            )
            continue
        _generate_samples(
            client=client,
            uses_v1=uses_v1,
            args=temp_args,
            call_params=call_params,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
