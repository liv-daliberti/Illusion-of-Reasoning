#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style crossword inference for Azure-hosted models.

Behavior:
- Single-pass generation with the standard crossword system prompt.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are
  generated.
- Uses Azure OpenAI (Responses API if available; falls back to Chat
  Completions).

This script mirrors the Azure math gateway runner but targets cryptic
crosswords instead.
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
from src.inference.gateways.base import setup_gateway_logger
from src.inference.utils import gateway_utils as _gateway_utils
from src.inference.utils.math_pass_utils import (
    build_second_pass_cue_strings,
    contains_canon as _contains_canon,
    extract_blocks as _extract_blocks,
    valid_tag_structure as _valid_tag_structure,
)
from src.inference.utils.task_registry import CROSSWORD_SYSTEM_PROMPT


logger = setup_gateway_logger(__name__)

# RuntimeError("Empty model output ...") can still bubble after retries.
_EMPTY_OUTPUT_MARKER = "Empty model output"

# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = CROSSWORD_SYSTEM_PROMPT
PASS2_KEYS = ("pass2a", "pass2b", "pass2c", "pass2d")

# Bind common helpers with defensive fallbacks for stubbed environments.
append_jsonl_row = _gateway_utils.append_jsonl_row
build_math_gateway_arg_parser = _gateway_utils.build_math_gateway_arg_parser
build_usage_dict = _gateway_utils.build_usage_dict
build_retry_context = _gateway_utils.build_retry_context
call_with_gateway_retries = _gateway_utils.call_with_gateway_retries
call_with_gateway_retries_compat = _gateway_utils.call_with_gateway_retries_compat
iter_jsonl_objects = _gateway_utils.iter_jsonl_objects
limit_dataset_examples = _gateway_utils.limit_dataset_examples
load_local_json_dataset = _gateway_utils.load_local_json_dataset
load_remote_dataset_default = _gateway_utils.load_remote_dataset_default
locked_file = _gateway_utils.locked_file
resolve_output_dir_for_temperature = _gateway_utils.resolve_output_dir_for_temperature
scan_existing_problem_samples = _gateway_utils.scan_existing_problem_samples
setup_hf_cache_dir_env = _gateway_utils.setup_hf_cache_dir_env

GatewayCallParams = _gateway_utils.GatewayCallParams
AzureCallParams = GatewayCallParams


# Crossword-friendly canon: casefold; strip spaces, hyphens, punctuation.
RE_PUNCT = re.compile(r"[^a-z0-9]", re.I)


def _canon_cross(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    lowered = text.strip().lower()
    lowered = lowered.replace("–", "-").replace("—", "-")
    return RE_PUNCT.sub("", lowered)


def _format_clue(clue: str, enumeration: Optional[str]) -> str:
    enum_text = f" ({enumeration})" if enumeration else ""
    return f"Clue: {clue}{enum_text}"


def _extract_fields(example: Dict[str, Any]) -> Tuple[Optional[str], Any, Optional[str]]:
    clue = (
        example.get("clue")
        or example.get("problem")
        or example.get("question")
        or example.get("prompt")
        or example.get("instruction")
        or example.get("query")
    )
    answer = (
        example.get("answer")
        or example.get("solution")
        or example.get("final_answer")
        or example.get("boxed_answer")
        or example.get("target")
    )
    enumeration = example.get("enumeration") or example.get("enum") or example.get("lengths")
    if isinstance(enumeration, (list, tuple)):
        enumeration = " ".join(str(part) for part in enumeration)
    return clue, answer, enumeration


def _iter_crossword_samples(
    dataset,
    num_samples: int,
    existing: Dict[str, set[int]],
) -> Iterable[Tuple[str, Any, Optional[str], int]]:
    for example in dataset:
        clue, gold_answer, enumeration = _extract_fields(example)
        if not clue or gold_answer is None:
            continue
        generated_indices = existing.get(clue, set())
        for sample_idx in range(num_samples):
            if sample_idx in generated_indices:
                continue
            yield clue, gold_answer, enumeration, sample_idx


def _prepare_dataset(args: argparse.Namespace, outpath: str):
    dataset_id = (args.dataset_id or "").upper()
    cache_dir = setup_hf_cache_dir_env("./.hf_cache")

    if dataset_id in ("CROSSWORD-LOCAL", "CROSSWORD_LOCAL") or args.dataset_path:
        if not args.dataset_path:
            raise RuntimeError("CROSSWORD-LOCAL requires --dataset_path.")
        dataset = load_local_json_dataset(args.dataset_path)
        dataset_name = os.path.basename(args.dataset_path)
    else:
        dataset = load_remote_dataset_default(args.dataset_id, split=args.split, cache_dir=cache_dir)
        dataset_name = args.dataset_id

    dataset = limit_dataset_examples(
        dataset,
        args.num_examples,
        from_end=getattr(args, "examples_from_end", False),
        start=getattr(args, "dataset_start", 0),
    )
    dataset = dataset.shuffle(seed=args.seed)
    existing = scan_existing_problem_samples(outpath)
    logger.info(
        "Dataset: %s split=%s | N=%d | existing=%d",
        dataset_name,
        args.split,
        len(dataset),
        len(existing),
    )
    logger.info("Output: %s", outpath)
    return dataset, existing


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
    """
    Bundled inputs needed to construct a JSONL result row.
    """

    clue: str
    gold_answer: Any
    enumeration: Optional[str]
    sample_idx: int
    text: str
    finish_reason: Any
    usage: Any


# ----------------------- Azure client + call -----------------------
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


def _call_model(
    client,
    uses_v1: bool,
    deployment: str,
    clue: str,
    enumeration: Optional[str],
    params: AzureCallParams,
):
    if uses_v1 and hasattr(client, "responses"):
        resp = client.responses.create(
            model=deployment,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": _format_clue(clue, enumeration)}],
            temperature=params.temperature,
            top_p=params.top_p,
            max_output_tokens=params.max_output_tokens,
            timeout=params.request_timeout,
        )
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
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_clue(clue, enumeration)},
        ],
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


def _call_model_with_messages(
    client,
    uses_v1: bool,
    deployment: str,
    messages: list[dict[str, str]],
    params: AzureCallParams,
):
    if uses_v1 and hasattr(client, "responses"):
        input_messages = [msg for msg in messages if msg.get("role") != "system"]
        resp = client.responses.create(
            model=deployment,
            instructions=SYSTEM_PROMPT,
            input=input_messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_output_tokens=params.max_output_tokens,
            timeout=params.request_timeout,
        )
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
    clue: str,
    enumeration: Optional[str],
    prev_output: str,
    cue: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_clue(clue, enumeration)},
        {"role": "assistant", "content": prev_output},
        {"role": "user", "content": cue},
    ]


def _build_pass_dict(
    *,
    text: str,
    gold_answer_canon: Optional[str],
    finish_reason: Any,
    usage: Any,
    enumeration: Optional[str],
    prev_output: Optional[str] = None,
    cue_text: Optional[str] = None,
    injected_cue: bool = False,
) -> Dict[str, Any]:
    _, ans = _extract_blocks(text)
    pred_canon = _canon_cross(ans)
    is_correct = _contains_canon(pred_canon, gold_answer_canon)
    pass_dict: Dict[str, Any] = {
        "output": text.strip(),
        "pred_answer": ans,
        "pred_answer_canon": pred_canon,
        "is_correct_pred": is_correct,
        "valid_tag_structure": _valid_tag_structure(text),
        "finish_reason": finish_reason,
    }
    if enumeration is not None:
        pass_dict["enumeration"] = enumeration
    if prev_output is not None:
        pass_dict["prev_output"] = prev_output
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

        gold_answer_canon = record.get("gold_answer_canon")
        enumeration = record.get("enumeration")
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
            messages = _build_pass2_messages(
                record.get("problem", ""),
                enumeration if isinstance(enumeration, str) else None,
                prev_output.strip(),
                cue_text,
            )
            call_fn = partial(
                _call_model_with_messages,
                client,
                uses_v1,
                args.deployment,
                messages,
                call_params,
            )
            try:
                call_result2 = _call_with_retries_compat(
                    call_fn,
                    args,
                    record.get("sample_idx", -1),
                    f"{record.get('problem', '')} | cue{idx + 1}",
                )
            except RuntimeError as exc:
                if _is_empty_output_error(exc):
                    logger.warning(
                        "Skipping empty pass2 output after retries (id=%s cue=%s).",
                        record.get("sample_idx", -1),
                        key,
                    )
                    continue
                raise
            pass2_results[key] = _build_pass_dict(
                text=call_result2[0],
                gold_answer_canon=gold_answer_canon,
                finish_reason=call_result2[1],
                usage=call_result2[2],
                enumeration=enumeration if isinstance(enumeration, str) else None,
                prev_output=prev_output.strip(),
                cue_text=cue_text,
                injected_cue=injected,
            )
        if pass2_results:
            if _update_record_with_lock(
                output_path,
                problem=record.get("problem", ""),
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
    sample_iter = _iter_crossword_samples(dataset, args.num_samples, existing)
    for clue, gold_answer, enumeration, sample_idx in sample_iter:
        call_fn = partial(
            _call_model,
            client,
            uses_v1,
            args.deployment,
            clue,
            enumeration,
            call_params,
        )
        try:
            call_result = _call_with_retries_compat(call_fn, args, sample_idx, clue)
        except RuntimeError as exc:
            if _is_empty_output_error(exc):
                logger.warning(
                    "Skipping empty pass1 output after retries (sample_idx=%s clue=%s).",
                    sample_idx,
                    clue,
                )
                continue
            raise

        row = _build_result_row(
            clue=clue,
            gold_answer=gold_answer,
            enumeration=enumeration,
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
            for idx, cue in enumerate(cue_strs):
                if idx >= len(PASS2_KEYS):
                    break
                is_neutral = total_cues >= 4 and idx == total_cues - 1
                injected = not is_neutral
                cue_text = cue.strip()
                messages = _build_pass2_messages(
                    clue,
                    enumeration,
                    str(call_result[0]).strip(),
                    cue_text,
                )
                call_fn = partial(
                    _call_model_with_messages,
                    client,
                    uses_v1,
                    args.deployment,
                    messages,
                    call_params,
                )
                try:
                    call_result2 = _call_with_retries_compat(
                        call_fn,
                        args,
                        sample_idx,
                        f"{clue} | cue{idx + 1}",
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
                pass2_results[PASS2_KEYS[idx]] = _build_pass_dict(
                    text=call_result2[0],
                    gold_answer_canon=row["gold_answer_canon"],
                    finish_reason=call_result2[1],
                    usage=call_result2[2],
                    enumeration=enumeration,
                    prev_output=str(call_result[0]).strip(),
                    cue_text=cue_text,
                    injected_cue=injected,
                )
            for key, pass_dict in pass2_results.items():
                row[key] = pass_dict
            if "pass2c" in pass2_results:
                row["pass2"] = pass2_results["pass2c"]
            elif pass2_results:
                row["pass2"] = pass2_results[PASS2_KEYS[len(pass2_results) - 1]]

        append_jsonl_row(output_path, row)
        total_new += 1
        existing.setdefault(clue, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples -> %s", total_new, output_path)


def _build_result_row(
    result: AzureResultRowInput | None = None,
    args: argparse.Namespace | None = None,
    call_params: AzureCallParams | None = None,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    if result is None:
        required_keys = ("clue", "gold_answer", "sample_idx", "text")
        missing = [key for key in required_keys if key not in legacy_kwargs]
        if missing:
            raise TypeError(f"_build_result_row() missing required arguments: {', '.join(missing)}")
        result = AzureResultRowInput(
            clue=legacy_kwargs["clue"],
            gold_answer=legacy_kwargs["gold_answer"],
            enumeration=legacy_kwargs.get("enumeration"),
            sample_idx=legacy_kwargs["sample_idx"],
            text=legacy_kwargs["text"],
            finish_reason=legacy_kwargs.get("finish_reason"),
            usage=legacy_kwargs.get("usage"),
        )
    if args is None or call_params is None:
        raise TypeError("_build_result_row() requires args and call_params")

    canon_gold = _canon_cross(str(result.gold_answer) if result.gold_answer is not None else None)
    _, ans = _extract_blocks(result.text)
    pred_canon = _canon_cross(ans)
    is_correct = _contains_canon(pred_canon, canon_gold)

    row: Dict[str, Any] = {
        "problem": result.clue,
        "gold_answer": result.gold_answer,
        "gold_answer_canon": canon_gold,
        "split": args.split,
        "step": args.step,
        "sample_idx": result.sample_idx,
    }
    if result.enumeration is not None:
        row["enumeration"] = result.enumeration

    row.update(
        {
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "api_version": args.api_version,
            "temperature": call_params.temperature,
            "top_p": call_params.top_p,
            "pass1": _build_pass_dict(
                text=result.text,
                gold_answer_canon=canon_gold,
                finish_reason=result.finish_reason,
                usage=None,
                enumeration=result.enumeration,
                prev_output=None,
            ),
        },
    )

    if result.usage:
        row["usage"] = build_usage_dict(result.usage)

    return row


def _parse_args() -> argparse.Namespace:
    default_cfg = load_azure_config()
    parser = build_math_gateway_arg_parser(
        default_temperature=0.7,
        description="Azure crossword gateway runner.",
    )
    parser.set_defaults(dataset_id="CROSSWORD-LOCAL")

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
