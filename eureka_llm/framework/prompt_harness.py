"""Prompt harness helpers for concise, contract-based agent prompting."""

from __future__ import annotations

import json
from hashlib import sha256


def build_contract_block(agent: str, objective: str, required_outputs: list[str], hard_constraints: list[str]) -> str:
    """Return a compact prompt contract block with stable structure."""
    payload = {
        "agent": agent,
        "objective": objective,
        "required_outputs": required_outputs,
        "hard_constraints": hard_constraints,
    }
    contract_id = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    lines = [
        f"## Prompt Contract [{contract_id}]",
        f"- Agent: {agent}",
        f"- Objective: {objective}",
        "- Required outputs:",
    ]
    lines.extend([f"  - {item}" for item in required_outputs])
    lines.append("- Hard constraints:")
    lines.extend([f"  - {item}" for item in hard_constraints])
    return "\n".join(lines)

