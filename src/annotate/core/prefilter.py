"""Shared regex utilities for shift-cue detection and tag extraction."""

import re
from typing import List, Optional, Tuple


RE_THINK = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Strict cue list (keep only explicit, high-precision markers)
SHIFT_CAND_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # ───────── Explicit self-interruptions / rethinks ─────────
    ("wait", re.compile(r"(?i)(?:^|\W)wait(?:\W|$)")),
    ("hold on", re.compile(r"(?i)\bhold (?:on|up)\b")),
    ("hang on", re.compile(r"(?i)\bhang on\b")),
    ("on second thought", re.compile(r"(?i)\bon (?:second|further) thought\b")),
    ("reconsider", re.compile(r"(?i)\breconsider\b")),
    ("rethink", re.compile(r"(?i)\bre-?think(?:ing)?\b")),
    ("scratch that", re.compile(r"(?i)\bscratch that\b")),
    ("strike that", re.compile(r"(?i)\bstrike that\b")),
    ("I take that back", re.compile(r"(?i)\bI take (?:that|it) back\b")),
    ("I retract", re.compile(r"(?i)\bI retract\b")),
    # ───────── Explicit corrections / replacements ─────────
    ("let me correct", re.compile(r"(?i)\blet'?s? (?:correct|fix) (?:that|this)\b")),
    ("correction_keyword", re.compile(r"(?i)\bcorrection\b")),
    ("correction_colon", re.compile(r"(?i)\bcorrection:\b")),
    ("to correct", re.compile(r"(?i)\bto correct\b")),
    ("fix that", re.compile(r"(?i)\bfix (?:that|this)\b")),
    ("change to", re.compile(r"(?i)\bchange (?:that|this)?\s*to\b")),
    ("switch to", re.compile(r"(?i)\bswitch (?:to|over)\b")),
    ("replace with", re.compile(r"(?i)\breplace (?:it|that|this)?\s*with\b")),
    ("try instead", re.compile(r"(?i)\btry (?:this|that )?instead\b")),
    ("consider instead", re.compile(r"(?i)\bconsider (?:instead|alternatively)\b")),
    # ───────── Admissions of error ─────────
    ("wrong_self", re.compile(r"(?i)\bi (?:was|am) wrong\b")),
    ("wrong_generic", re.compile(r"(?i)\bthat(?:'s| is)? wrong\b")),
    ("incorrect", re.compile(r"(?i)\bincorrect\b|\bnot correct\b")),
    ("mistake_generic", re.compile(r"(?i)\b(?:my )?mistake\b|\bI made a mistake\b")),
    ("misread", re.compile(r"(?i)\bmis-?read\b|\bI misread\b")),
    ("miscount", re.compile(r"(?i)\bmis-?count(?:ed|ing)?\b")),
    ("miscalc", re.compile(r"(?i)\bmis-?calculat(?:e|ed|ion)\b|\bcalc(?:ulation)? error\b")),
    ("misapply", re.compile(r"(?i)\bmis-?appl(?:y|ied|ication)\b")),
    ("misparse", re.compile(r"(?i)\bmis-?pars(?:e|ed|ing)\b")),
    ("misspell", re.compile(r"(?i)\bmis-?spell(?:ed|ing)?\b|\bmisspelt\b|\bmisspelled\b")),
    ("misindex", re.compile(r"(?i)\bmis-?index(?:ed|ing)?\b")),
    ("misuse_rule", re.compile(r"(?i)\bmis-?us(?:e|ed|ing)\b")),
    ("typo", re.compile(r"(?i)\btypo\b")),
    ("off by one", re.compile(r"(?i)\boff[-\s]?by[-\s]?one\b")),
    # ───────── Constraint/length/pattern mismatch (xword-friendly) ─────────
    ("doesnt fit", re.compile(r"(?i)\bdoes(?:n'?t| not) (?:fit|match)(?: length| pattern)?\b")),
    ("letters dont fit", re.compile(r"(?i)\bletters? do(?:es)?n'?t (?:fit|match)\b")),
    ("pattern mismatch", re.compile(r"(?i)\bpattern (?:mis)?match\b")),
    ("length mismatch", re.compile(r"(?i)\blength (?:mis)?match\b")),
    ("too many letters", re.compile(r"(?i)\btoo many letters\b")),
    ("too few letters", re.compile(r"(?i)\b(?:not enough|too few) letters\b")),
    ("wrong length", re.compile(r"(?i)\bwrong length\b")),
    ("violates enumeration", re.compile(r"(?i)\bviolates? (?:the )?enumeration\b")),
    ("doesnt parse", re.compile(r"(?i)\bdoes(?:n'?t| not) parse\b")),
    ("definition mismatch", re.compile(r"(?i)\bdefinition (?:doesn'?t|does not) match\b")),
    ("anagram doesnt work", re.compile(r"(?i)\banagram (?:doesn'?t|does not) (?:work|fit)\b")),
    ("fodder mismatch", re.compile(r"(?i)\bfodder (?:doesn'?t|does not) (?:match|fit)\b")),
    # ───────── Logical contradiction / impossibility ─────────
    ("contradiction", re.compile(r"(?i)\bcontradict(?:s|ion|ory)\b")),
    ("inconsistent", re.compile(r"(?i)\binconsistent\b")),
    ("cant be", re.compile(r"(?i)\bcan'?t be\b|\bcannot be\b")),
    ("impossible", re.compile(r"(?i)\bimpossible\b")),
]


def extract_think(txt: str) -> Optional[str]:
    """Return the <think> block contents, if present."""
    match = RE_THINK.search(txt or "")
    return match.group(1).strip() if match else None


def find_shift_cues(think: str) -> Tuple[List[str], Optional[int]]:
    """Find cue names and the earliest character index in the think text."""
    if not think:
        return [], None
    hits: List[str] = []
    first_pos = None
    for name, pat in SHIFT_CAND_PATTERNS:
        match = pat.search(think)
        if match:
            hits.append(name)
            pos = match.start()
            if first_pos is None or pos < first_pos:
                first_pos = pos
    return hits, first_pos


def find_cue_hits(think: str) -> List[str]:
    """Return only the names of cues found in think text."""
    return find_shift_cues(think)[0]


__all__ = [
    "RE_THINK",
    "RE_ANSWER",
    "SHIFT_CAND_PATTERNS",
    "extract_think",
    "find_shift_cues",
    "find_cue_hits",
]
