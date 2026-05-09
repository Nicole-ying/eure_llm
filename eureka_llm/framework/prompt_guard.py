"""prompt_guard.py — Phase-3 programmable guardrails for zero-shot outputs."""

from __future__ import annotations
import re


ENV_LEAK_PATTERNS = [
    r"\bmountaincar\b", r"\blunarlander\b", r"\bbipedalwalker\b",
    r"\bhalfcheetah\b", r"\bhumanoid\b", r"\bant\b",
]

ABS_THRESHOLD_PATTERNS = [
    r"[<>]=?\s*\d+(\.\d+)?",
    r"\bexactly\s+\d+(\.\d+)?\b",
    r"\bfinal entropy\s*<\s*0\.\d+\b",
]

IMPLICIT_ENV_HINT_PATTERNS = [
    r"\bhill climb\b", r"\blanding pad\b", r"\bleg contact\b",
    r"\bhull angle\b", r"\bbox2d\b", r"\bbiped\b", r"\bthruster\b",
]


def validate_zero_shot_output(text: str) -> dict:
    t = (text or "").lower()
    env_hits = [p for p in ENV_LEAK_PATTERNS if re.search(p, t)]
    abs_hits = [p for p in ABS_THRESHOLD_PATTERNS if re.search(p, t)]
    implicit_hits = [p for p in IMPLICIT_ENV_HINT_PATTERNS if re.search(p, t)]
    return {
        "env_leakage": bool(env_hits),
        "implicit_env_hint": bool(implicit_hits),
        "absolute_threshold_language": bool(abs_hits),
        "env_hit_patterns": env_hits,
        "implicit_hit_patterns": implicit_hits,
        "absolute_hit_patterns": abs_hits,
        "passed": not env_hits and not abs_hits and not implicit_hits,
    }
