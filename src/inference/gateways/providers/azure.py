#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style MATH-500 inference for Azure-hosted DeepSeek-R1 (open-source).

Behavior:
- Single-pass generation with the same math system prompt used in GRPO runs.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are
  generated.
- Uses Azure OpenAI (Responses API if available; falls back to Chat
  Completions).

This script is a **specialized / legacy gateway runner**. For canonical
HF-based math evaluation, prefer ``src.inference.cli.unified_math`` using local
checkpoints.

Usage (env-based auth):
  export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
  export AZURE_OPENAI_API_KEY="***"
  export AZURE_OPENAI_DEPLOYMENT="deepseek-r1"  # or your deployment name
  python -m src.inference.gateways.providers.azure \
      --output_dir artifacts/results/deepseek-r1
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Dict

from src.annotate import build_preferred_client, load_azure_config
from src.inference.domains.math.math_core import load_math500 as _load_math500_core
from src.inference.gateways.base import get_task_spec, setup_gateway_logger
from src.inference.utils import common as _common_utils
from src.inference.utils.gateway_utils import GatewayCallParams
from src.inference.utils.gateway_utils import RetryContext as GatewayRetryContext
from src.inference.utils.gateway_utils import build_retry_context, call_with_gateway_retries_compat
from src.inference.utils.task_registry import MATH_SYSTEM_PROMPT


if TYPE_CHECKING:
    pass


logger = setup_gateway_logger(__name__)

TASK_SPEC = get_task_spec("math-azure")

# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT
PASS2_KEYS = ("pass2a", "pass2b", "pass2c", "pass2d")

# Bind common helpers with defensive fallbacks for stubbed environments.
append_jsonl_row = _common_utils.append_jsonl_row
build_math_gateway_arg_parser = _common_utils.build_math_gateway_arg_parser
build_math_gateway_row_base = _common_utils.build_math_gateway_row_base
build_second_pass_cue_strings = _common_utils.build_second_pass_cue_strings
build_usage_dict = _common_utils.build_usage_dict
call_with_gateway_retries = _common_utils.call_with_gateway_retries
iter_jsonl_objects = _common_utils.iter_jsonl_objects
locked_file = _common_utils.locked_file
resolve_output_dir_for_temperature = _common_utils.resolve_output_dir_for_temperature
RetryContext = getattr(_common_utils, "RetryContext", GatewayRetryContext)
GatewayCallParams = getattr(_common_utils, "GatewayCallParams", GatewayCallParams)
AzureCallParams = GatewayCallParams
_canon_math = _common_utils.canon_math
_extract_blocks = _common_utils.extract_blocks
iter_math_gateway_samples = _common_utils.iter_math_gateway_samples
load_remote_dataset_default = _common_utils.load_remote_dataset_default
prepare_math_gateway_dataset_from_args = _common_utils.prepare_math_gateway_dataset_from_args
_valid_tag_structure = _common_utils.valid_tag_structure


def _call_with_retries_compat(func, args: argparse.Namespace, sample_idx: int, problem: str):
    """
    Invoke ``call_with_gateway_retries`` while tolerating legacy signatures used in tests.
    """
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
        # Fall back to versions that expect keyword args instead of a context object.
        return call_with_gateway_retries(func, args=args, context=retry_ctx)


def load_math500(
    cache_dir: str,
    split: str,
    seed: int,
    dataset_path: str | None = None,
):
    """Thin wrapper that defers to the shared math-500 loader (monkeypatchable in tests)."""
    return _load_math500_core(cache_dir, split, seed, dataset_path)


@dataclass
class AzureResultRowInput:
    """
    Bundled inputs needed to construct a JSONL result row.

    :param problem: Normalized problem text for the example.
    :param gold_answer: Ground-truth answer associated with the problem.
    :param sample_idx: Sample index for this generation.
    :param text: Raw model output text.
    :param finish_reason: Finish reason reported by the Azure API.
    :param usage: Optional usage object returned by the Azure SDK.
    """

    problem: str
    gold_answer: Any
    sample_idx: int
    text: str
    finish_reason: Any
    usage: Any


# ----------------------- Azure client + call -----------------------
def _make_client(args: argparse.Namespace):
    """
    Construct an Azure/OpenAI client and resolve endpoint/deployment.

    :param args: Parsed CLI arguments that may override endpoint and deployment.
    :returns: Tuple ``(client, uses_v1, endpoint, deployment, api_version)`` where
        ``uses_v1`` indicates whether the Responses API is preferred.
    :raises RuntimeError: If an API key cannot be resolved from arguments or environment.
    """
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
    problem: str,
    params: AzureCallParams,
):
    """
    Call the Azure DeepSeek deployment and return text, finish reason, and usage.

    :param client: Azure/OpenAI client created by :func:`_make_client`.
    :param uses_v1: Whether to prefer the Responses API (v1) over Chat Completions.
    :param deployment: Name of the Azure deployment to use.
    :param problem: Raw problem text to send to the model.
    :param params: Generation parameters such as temperature and max tokens.
    :returns: Tuple ``(text, finish_reason, usage)`` describing the model response.
    """
    if uses_v1 and hasattr(client, "responses"):
        resp = client.responses.create(
            model=deployment,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": problem}],
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

    # Legacy Chat Completions
    resp = client.chat.completions.create(
        model=deployment,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_output_tokens,
        timeout=params.request_timeout,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
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
    """
    Call the Azure deployment with an explicit chat message list.
    """
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
            call_fn = partial(
                _call_model_with_messages,
                client,
                uses_v1,
                args.deployment,
                messages,
                call_params,
            )
            call_result2 = _call_with_retries_compat(
                call_fn,
                args,
                record.get("sample_idx", -1),
                f"{record.get('problem', '')} | cue{idx + 1}",
            )
            pass2_results[key] = _build_pass_dict(
                text=call_result2[0],
                gold_answer_canon=gold_answer_canon,
                finish_reason=call_result2[1],
                usage=call_result2[2],
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


def _prepare_dataset(args: argparse.Namespace, outpath: str):
    """
    Load and shuffle the dataset, applying resume/fill logic.

    :param args: Parsed CLI arguments controlling dataset choice and sampling.
    :param outpath: Output path used to determine which samples already exist.
    :returns: Tuple ``(dataset, existing)`` where ``existing`` maps problems
        to the set of already-filled sample indices.
    """
    dataset, existing, _ = prepare_math_gateway_dataset_from_args(
        args=args,
        outpath=outpath,
        logger=logger,
        load_math500_fn=load_math500,
        load_remote_dataset_fn=load_remote_dataset_default,
        cache_dir=os.path.abspath("./.hf_cache"),
    )
    return dataset, existing


def _generate_samples(
    client,
    uses_v1: bool,
    args: argparse.Namespace,
    call_params: AzureCallParams,
    output_path: str,
) -> None:
    """
    Main generation loop over the dataset.

    :param client: Azure/OpenAI client created by :func:`_make_client`.
    :param uses_v1: Whether to use the Responses API for generation.
    :param args: Parsed CLI arguments controlling generation behavior.
    :param call_params: Azure generation parameters such as temperature and limits.
    :param output_path: Path to the JSONL file where results are written.
    :returns: ``None``. Samples are written to disk and progress is logged.
    """
    dataset, existing = _prepare_dataset(args, output_path)

    total_new = 0
    sample_iter = iter_math_gateway_samples(dataset, args.num_samples, existing)
    for problem, gold_answer, sample_idx in sample_iter:
        call_fn = partial(
            _call_model,
            client,
            uses_v1,
            args.deployment,
            problem,
            call_params,
        )
        call_result = _call_with_retries_compat(call_fn, args, sample_idx, problem)

        row = _build_result_row(
            problem=problem,
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
            for idx, cue in enumerate(cue_strs):
                if idx >= len(PASS2_KEYS):
                    break
                is_neutral = total_cues >= 4 and idx == total_cues - 1
                injected = not is_neutral
                cue_text = cue.strip()
                messages = _build_pass2_messages(problem, str(call_result[0]).strip(), cue_text)
                call_fn = partial(
                    _call_model_with_messages,
                    client,
                    uses_v1,
                    args.deployment,
                    messages,
                    call_params,
                )
                call_result2 = _call_with_retries_compat(call_fn, args, sample_idx, f"{problem} | cue{idx + 1}")
                pass2_results[PASS2_KEYS[idx]] = _build_pass_dict(
                    text=call_result2[0],
                    gold_answer_canon=row["gold_answer_canon"],
                    finish_reason=call_result2[1],
                    usage=call_result2[2],
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
        existing.setdefault(problem, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples → %s", total_new, output_path)


def _build_result_row(
    result: AzureResultRowInput | None = None,
    args: argparse.Namespace | None = None,
    call_params: AzureCallParams | None = None,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    """
    Build a JSONL row for a single generated sample.

    Accepts either an :class:`AzureResultRowInput` bundle or legacy keyword
    arguments (``problem``, ``gold_answer``, ``sample_idx``, ``text``,
    ``finish_reason``, ``usage``) for backwards compatibility in tests.

    :param result: Inputs captured for a single generated sample.
    :param args: Parsed CLI arguments (used for metadata fields).
    :param call_params: Azure generation parameters used for the call.
    :returns: Dictionary representing one JSONL row.
    """
    if result is None:
        required_keys = ("problem", "gold_answer", "sample_idx", "text")
        missing = [key for key in required_keys if key not in legacy_kwargs]
        if missing:
            raise TypeError(f"_build_result_row() missing required arguments: {', '.join(missing)}")
        result = AzureResultRowInput(
            problem=legacy_kwargs["problem"],
            gold_answer=legacy_kwargs["gold_answer"],
            sample_idx=legacy_kwargs["sample_idx"],
            text=legacy_kwargs["text"],
            finish_reason=legacy_kwargs.get("finish_reason"),
            usage=legacy_kwargs.get("usage"),
        )
    if args is None or call_params is None:
        raise TypeError("_build_result_row() requires args and call_params")

    canon_gold = _canon_math(result.gold_answer)
    _, ans = _extract_blocks(result.text)
    pred_canon = _canon_math(ans)
    is_correct = bool(pred_canon and canon_gold and canon_gold in pred_canon)

    row = build_math_gateway_row_base(
        problem=result.problem,
        gold_answer=result.gold_answer,
        gold_answer_canon=canon_gold,
        split=args.split,
        step=args.step,
        sample_idx=result.sample_idx,
    )
    row.update(
        {
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "api_version": args.api_version,
            "temperature": call_params.temperature,
            "top_p": call_params.top_p,
            "pass1": {
                "output": result.text.strip(),
                "pred_answer": ans,
                "pred_answer_canon": pred_canon,
                "is_correct_pred": is_correct,
                "valid_tag_structure": _valid_tag_structure(result.text),
                "finish_reason": result.finish_reason,
            },
        },
    )

    if result.usage:
        row["usage"] = build_usage_dict(result.usage)

    return row


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the Azure DeepSeek MATH-500 runner.

    :returns: Parsed :class:`argparse.Namespace` with configuration values.
    """
    default_cfg = load_azure_config()
    parser = build_math_gateway_arg_parser(
        default_temperature=0.7,
        description="Azure DeepSeek-R1 MATH-500 runner.",
    )

    # Azure params (env defaults from configs/azure.yml)
    parser.add_argument("--endpoint", default=default_cfg.get("endpoint"))
    parser.add_argument(
        "--deployment",
        default=None,
        help="Azure deployment name (e.g., deepseek-r1).",
    )
    parser.add_argument("--api_version", default=default_cfg.get("api_version"))
    parser.add_argument("--api_key", default=None)
    parser.add_argument(
        "--use_v1",
        type=int,
        default=int(default_cfg.get("use_v1", 1)),
        help="1 → prefer Responses API (v1); 0 → force Chat Completions.",
    )

    # Retry controls
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff", type=float, default=2.0)

    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint for single-pass Azure DeepSeek MATH-500 inference.

    :returns: ``None``. The function parses arguments and runs the generation loop.
    """
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

    # Normalise args with resolved endpoint values for logging and rows.
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


__all__ = ["load_math500", "main"]


if __name__ == "__main__":
    main()
