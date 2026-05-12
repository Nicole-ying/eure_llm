"""Context engineering helpers: concise evidence packets for agent prompts."""

from __future__ import annotations

import json


def build_evidence_packet(
    diagnostics: dict | None = None,
    must_fix_issues: list[str] | None = None,
    mandatory_lessons: list[str] | None = None,
    max_items: int = 6,
    strategy: str = "balanced",
) -> str:
    if strategy not in {"balanced", "strict"}:
        strategy = "balanced"
    if strategy == "strict":
        max_items = min(max_items, 3)
    payload = {
        "diagnostics": diagnostics or {},
        "must_fix_issues": (must_fix_issues or [])[:max_items],
        "mandatory_lessons": (mandatory_lessons or [])[:max_items],
    }
    return "## Evidence Packet\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"
