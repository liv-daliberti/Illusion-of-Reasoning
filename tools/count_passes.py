#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count pass1/pass2* coverage for gateway JSONL runs.

Usage:
  python tools/count_passes.py
  python tools/count_passes.py artifacts/results/gpt4o-math-portkey-temp03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


PASS_KEYS = ("pass1", "pass2a", "pass2b", "pass2c", "pass2d", "pass2")
COUNT_KEYS = ["rows", "pass1", "pass1_annotated", "pass1_output_empty"] + [
    key for key in PASS_KEYS if key != "pass1"
]


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


def has_pass(record: Dict, key: str) -> bool:
    value = record.get(key)
    return isinstance(value, dict) and bool(value)


def has_pass1_annotation(record: Dict) -> bool:
    pass1 = record.get("pass1")
    if not isinstance(pass1, dict):
        return False
    return bool(pass1.get("shift_rationale_gpt_model") or pass1.get("shift_rationale_gpt"))


def pass1_output_empty(record: Dict) -> bool:
    pass1 = record.get("pass1")
    if not isinstance(pass1, dict):
        return False
    output = pass1.get("output")
    if output is None:
        return True
    return not str(output).strip()


def gather_default_roots() -> List[Path]:
    patterns = [
        "gpt4o-math-portkey*",
        "gpt4o-xword-portkey*",
        "gpt4o-crossword-portkey*",
        "deepseek-r1-openrouter*",
        "deepseek-r1-openrouter-xword*",
        "deepseek-r1-openrouter-crossword*",
        "deepseek-r1-math-azure*",
        "gpt4o-math-azure*",
        "*-carpark-azure*",
        "*-rush-azure*",
        "*-xword-azure*",
        "*-crossword-azure*",
        "*-azure-xword*",
        "*-azure-crossword*",
        "gpt4o-xword-azure*",
        "gpt4o-azure-xword*",
        "deepseek-r1-xword-azure*",
        "deepseek-r1-azure-xword*",
    ]
    roots = {path for pattern in patterns for path in Path("artifacts/results").glob(pattern)}
    return sorted(roots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        help=(
            "One or more result roots (defaults to gpt4o-*-portkey*, deepseek-r1-openrouter*, and "
            "azure-run conventions including crossword variants)."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Only include JSONL files whose filename contains this substring (default: test).",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=None,
        help="Optional expected rows per root (e.g., 4000 for MATH-500 × 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.roots] if args.roots else gather_default_roots()

    if not roots:
        print("No roots found.")
        return

    table_rows: List[List[str]] = []
    header = ["root"] + COUNT_KEYS
    for root in roots:
        if not root.exists():
            continue
        files = [
            path
            for path in root.rglob("*.jsonl")
            if args.split is None or args.split in path.name
        ]
        if not files:
            print(f"\n{root}: no JSONL files found")
            continue

        total = 0
        counts = {key: 0 for key in PASS_KEYS}
        pass1_annotated = 0
        pass1_empty = 0
        for path in files:
            for record in iter_jsonl(path):
                total += 1
                for key in PASS_KEYS:
                    if has_pass(record, key):
                        counts[key] += 1
                if has_pass1_annotation(record):
                    pass1_annotated += 1
                if pass1_output_empty(record):
                    pass1_empty += 1

        row = [
            str(root),
            str(total),
            str(counts["pass1"]),
            str(pass1_annotated),
            str(pass1_empty),
        ] + [str(counts[key]) for key in PASS_KEYS if key != "pass1"]
        table_rows.append(row)

    if not table_rows:
        print("No matching JSONL files found.")
        return

    totals = [0] * (len(header) - 1)
    for row in table_rows:
        for idx in range(1, len(header)):
            try:
                totals[idx - 1] += int(row[idx])
            except ValueError:
                pass
    total_row = ["TOTAL"] + [str(value) for value in totals]
    base_total = totals[0]
    if args.expected_rows is not None:
        base_total = int(args.expected_rows) * len(table_rows)
    expected_cols = {"rows", "pass1", "pass1_annotated", "pass2a", "pass2b", "pass2c", "pass2d", "pass2"}
    missing_vals = []
    for key, value in zip(COUNT_KEYS, totals):
        if key in expected_cols:
            missing_vals.append(str(max(0, base_total - value)))
        else:
            missing_vals.append("-")
    missing_row = ["MISSING"] + [str(value) for value in missing_vals]

    widths = [len(col) for col in header]
    for row in table_rows + [total_row, missing_row]:
        widths = [max(widths[i], len(cell)) for i, cell in enumerate(row)]

    def fmt_row(items: List[str]) -> str:
        return " | ".join(item.ljust(widths[i]) for i, item in enumerate(items))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_row(header))
    print(sep)
    for row in table_rows:
        print(fmt_row(row))
    print(sep)
    print(fmt_row(total_row))
    print(fmt_row(missing_row))


if __name__ == "__main__":
    main()
