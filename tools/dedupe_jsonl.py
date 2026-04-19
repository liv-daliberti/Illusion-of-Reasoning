#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deduplicate JSONL rows by (problem, sample_idx, step, split).

Keeps the most complete row (highest pass coverage), falling back to the
latest row when scores are tied. Optionally trims to a max number of
samples per problem by dropping rows with sample_idx >= max_samples.

Usage:
  python tools/dedupe_jsonl.py path/to/file.jsonl --inplace
  python tools/dedupe_jsonl.py artifacts/results/gpt4o-math-portkey-temp07 --inplace
  python tools/dedupe_jsonl.py artifacts/results/gpt4o-math-portkey-temp07 --inplace --max-samples 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PASS_KEYS = ("pass1", "pass2a", "pass2b", "pass2c", "pass2d", "pass2")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.lstrip().startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def pass_score(record: Dict) -> Tuple[int, int]:
    score = 0
    for key in PASS_KEYS:
        value = record.get(key)
        if isinstance(value, dict) and value:
            score += 1
    output = ""
    pass1 = record.get("pass1") or {}
    if isinstance(pass1, dict):
        output = pass1.get("output") or ""
    return score, len(str(output))


def record_key(record: Dict) -> Optional[Tuple]:
    problem = (
        record.get("problem")
        or record.get("question")
        or record.get("prompt")
        or record.get("instruction")
        or record.get("query")
    )
    sample_idx = record.get("sample_idx")
    step = record.get("step")
    split = record.get("split")
    if problem is None or sample_idx is None:
        return None
    try:
        sample_idx = int(sample_idx)
    except (TypeError, ValueError):
        return None
    return (problem, sample_idx, step, split)


def dedupe_records(records: List[Dict], max_samples: Optional[int]) -> Tuple[List[Dict], int, int]:
    kept: Dict[Tuple, Dict] = {}
    order: List[Tuple] = []
    duplicates = 0
    for record in records:
        key = record_key(record)
        if key is None:
            order.append((id(record),))
            kept[(id(record),)] = record
            continue
        if key not in kept:
            kept[key] = record
            order.append(key)
            continue
        duplicates += 1
        prev = kept[key]
        if pass_score(record) >= pass_score(prev):
            kept[key] = record
    deduped = [kept[k] for k in order if k in kept]
    trimmed = 0
    if max_samples is not None and max_samples >= 0:
        filtered: List[Dict] = []
        for record in deduped:
            key = record_key(record)
            if key is None:
                filtered.append(record)
                continue
            sample_idx = key[1]
            if sample_idx < max_samples:
                filtered.append(record)
            else:
                trimmed += 1
        deduped = filtered
    return deduped, duplicates, trimmed


def write_records(path: Path, records: List[Dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
    os.replace(tmp_path, path)


def find_jsonl_files(root: Path, split_substr: Optional[str]) -> List[Path]:
    if root.is_file():
        return [root]
    files = [p for p in root.rglob("*.jsonl")]
    if split_substr is not None:
        files = [p for p in files if split_substr in p.name]
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JSONL file or directory root.")
    parser.add_argument(
        "--split",
        default="test",
        help="Only process JSONL files whose name contains this substring (default: test).",
    )
    parser.add_argument("--inplace", action="store_true", help="Rewrite the input file(s) in place.")
    parser.add_argument("--output", default=None, help="Output file (only valid when path is a single file).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Drop rows where sample_idx >= this value (e.g., 8 keeps sample_idx 0-7).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.path)
    files = find_jsonl_files(root, args.split)
    if not files:
        print("No JSONL files found.")
        return
    if args.output and len(files) != 1:
        raise SystemExit("--output is only supported when path is a single JSONL file.")
    for path in files:
        records = list(iter_jsonl(path))
        if not records:
            print(f"{path}: empty")
            continue
        deduped, duplicates, trimmed = dedupe_records(records, args.max_samples)
        out_path = Path(args.output) if args.output else path
        if args.inplace or args.output:
            write_records(out_path, deduped)
        print(
            f"{path}: rows={len(records)} unique={len(deduped)} "
            f"removed={duplicates} trimmed={trimmed}"
        )


if __name__ == "__main__":
    main()
