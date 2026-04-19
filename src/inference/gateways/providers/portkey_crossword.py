#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style crossword inference via the Portkey AI Gateway (e.g., Princeton AI
Sandbox).

Behavior:
- Single-pass generation with the standard crossword system prompt.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are
  generated.

This script mirrors the MATH Portkey gateway runner but targets cryptic
crosswords instead. Use it when you need Portkey/sandbox-based remote models.

Auth:
- Expects an API key in the `AI_SANDBOX_KEY` env var (Portkey key or sandbox
  key).

Example usage:
  export AI_SANDBOX_KEY="***"
  python -m src.inference.gateways.providers.portkey_crossword \
      --output_dir artifacts/results/gpt4o-xword-portkey \
      --dataset_id CROSSWORD-LOCAL \
      --dataset_path data/crossword/guardian_cryptonite_test.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, Optional, Tuple

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

# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = CROSSWORD_SYSTEM_PROMPT
PASS2_KEYS = ("pass2a", "pass2b", "pass2c", "pass2d")

# Bind gateway helpers.
append_jsonl_row = _gateway_utils.append_jsonl_row
build_math_gateway_arg_parser = _gateway_utils.build_math_gateway_arg_parser
build_usage_dict = _gateway_utils.build_usage_dict
iter_jsonl_objects = _gateway_utils.iter_jsonl_objects
limit_dataset_examples = _gateway_utils.limit_dataset_examples
load_local_json_dataset = _gateway_utils.load_local_json_dataset
load_remote_dataset_default = _gateway_utils.load_remote_dataset_default
locked_file = _gateway_utils.locked_file
resolve_output_dir_for_temperature = _gateway_utils.resolve_output_dir_for_temperature
scan_existing_problem_samples = _gateway_utils.scan_existing_problem_samples
setup_hf_cache_dir_env = _gateway_utils.setup_hf_cache_dir_env

PortkeyCallParams = _gateway_utils.GatewayCallParams


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
    return dataset, existing, dataset_name


# ----------------------- Portkey client + call -----------------------
@dataclass
class PortkeyRunConfig:
    """
    Configuration for a Portkey crossword pass.
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
    """Per-example metadata for a crossword row."""

    clue: str
    gold_answer: Any
    canon_gold: Optional[str]
    enumeration: Optional[str]
    sample_idx: int


@dataclass
class PortkeyCallResult:
    """Result of a single Portkey generation call."""

    text: str
    answer: str
    finish_reason: Any
    usage: Any


def _make_client():
    """
    Construct a Portkey client using ``AI_SANDBOX_KEY`` from the environment.
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
    clue: str,
    enumeration: Optional[str],
    params: PortkeyCallParams,
):
    """
    Call the Portkey model for a single crossword clue.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_clue(clue, enumeration)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_output_tokens,
        timeout=params.request_timeout,
    )
    text, finish_reason, usage = _gateway_utils.parse_openai_chat_response(resp)
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
    text, finish_reason, usage = _gateway_utils.parse_openai_chat_response(resp)
    if not text or not str(text).strip():
        raise RuntimeError(
            f"Empty model output (model={model}, finish_reason={finish_reason})"
        )
    return text, finish_reason, usage


def _build_pass2_messages(
    clue: str,
    enumeration: Optional[str],
    prev_output: str,
    cue: str,
) -> list[dict[str, str]]:
    """
    Build a chat history for second-pass generation with an injected cue.
    """
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
    """
    Build a pass dictionary for pass2 variants (or pass1 when cue_text is None).
    """
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
                enumeration=enumeration if isinstance(enumeration, str) else None,
                prev_output=prev_output.strip(),
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


def _build_portkey_row(
    example: ExampleContext,
    result: PortkeyCallResult,
    config: PortkeyRunConfig,
) -> Dict[str, Any]:
    """
    Build a single JSONL row for Portkey crossword inference and compute correctness.
    """
    pred_canon = _canon_cross(result.answer)
    is_correct = _contains_canon(pred_canon, example.canon_gold)

    row: Dict[str, Any] = {
        "problem": example.clue,
        "gold_answer": example.gold_answer,
        "gold_answer_canon": example.canon_gold,
        "split": config.split_name,
        "step": config.step,
        "sample_idx": example.sample_idx,
    }
    if example.enumeration is not None:
        row["enumeration"] = example.enumeration

    row.update(
        {
            "endpoint": "portkey-ai",
            "deployment": config.model_name,
            "api_version": None,
            "temperature": config.params.temperature,
            "top_p": config.params.top_p,
            "pass1": _build_pass_dict(
                text=result.text,
                gold_answer_canon=example.canon_gold,
                finish_reason=result.finish_reason,
                usage=None,
                enumeration=example.enumeration,
                prev_output=None,
            ),
        },
    )

    if result.usage is not None:
        row["usage"] = build_usage_dict(result.usage)

    return row


def run_portkey_crossword_inference(
    client,
    dataset,
    existing: Dict[str, set],
    config: PortkeyRunConfig,
) -> None:
    """
    Run single-pass crossword inference via Portkey and write JSONL results.
    """
    random.seed(config.seed)
    cue_strs = build_second_pass_cue_strings(config.second_pass_phrase) if config.two_pass else []
    total_cues = len(cue_strs)
    total_new = 0
    for clue, gold_answer, enumeration, sample_idx in _iter_crossword_samples(
        dataset,
        config.num_samples,
        existing,
    ):
        text, finish_reason, usage = _call_model(
            client=client,
            model=config.model_name,
            clue=clue,
            enumeration=enumeration,
            params=config.params,
        )

        _, answer = _extract_blocks(text)
        row = _build_portkey_row(
            ExampleContext(
                clue=clue,
                gold_answer=gold_answer,
                canon_gold=_canon_cross(str(gold_answer) if gold_answer is not None else None),
                enumeration=enumeration,
                sample_idx=sample_idx,
            ),
            PortkeyCallResult(
                text=text,
                answer=answer or "",
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
                messages = _build_pass2_messages(clue, enumeration, text.strip(), cue_text)
                text2, finish2, usage2 = _call_model_with_messages(
                    client=client,
                    model=config.model_name,
                    messages=messages,
                    params=config.params,
                )
                pass2_results[PASS2_KEYS[idx]] = _build_pass_dict(
                    text=text2,
                    gold_answer_canon=row["gold_answer_canon"],
                    finish_reason=finish2,
                    usage=usage2,
                    enumeration=enumeration,
                    prev_output=text.strip(),
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
        existing.setdefault(clue, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples → %s", total_new, config.output_path)


def main() -> None:
    """
    Parse arguments, load dataset, and run Portkey-based crossword inference.
    """
    parser = build_math_gateway_arg_parser(
        default_temperature=0.05,
        description="Portkey crossword runner.",
    )
    parser.set_defaults(dataset_id="CROSSWORD-LOCAL")
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

        dataset, existing, _ = _prepare_dataset(temp_args, output_path)
        logger.info("Running temperature=%s -> %s", temp_args.temperature, output_path)
        run_portkey_crossword_inference(
            client=client,
            dataset=dataset,
            existing=existing,
            config=config,
        )


if __name__ == "__main__":
    main()
