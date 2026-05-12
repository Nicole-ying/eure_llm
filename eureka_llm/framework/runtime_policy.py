"""Harness runtime policies for feedback-loop decisions."""

from __future__ import annotations

import json
from hashlib import sha256


def compute_evidence_fingerprint(validation_issues: list[str], proposal: dict) -> str:
    payload = {
        "validation_issues": validation_issues,
        "proposal_changed_count": proposal.get("changed_count", 0),
        "proposal_changes": proposal.get("proposed_changes", []),
    }
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def should_rerun_analyst(previous_fingerprint: str | None, current_fingerprint: str, strategy: str = "strict") -> bool:
    if strategy not in {"strict", "always_once"}:
        strategy = "strict"
    if strategy == "always_once":
        return True
    return previous_fingerprint != current_fingerprint
