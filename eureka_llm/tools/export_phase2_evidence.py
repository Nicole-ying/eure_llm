#!/usr/bin/env python3
"""Export Phase-2 evidence table from run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


def _round_dirs(run_dir: Path) -> list[Path]:
    items = [p for p in run_dir.glob("round*") if p.is_dir() and re.fullmatch(r"round\d+", p.name)]
    return sorted(items, key=lambda p: int(p.name.replace("round", "")))


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def export(run_dir: Path, out_csv: Path):
    rows = []
    for rd in _round_dirs(run_dir):
        conv = (rd / "analyst_conversation.json").read_text("utf-8") if (rd / "analyst_conversation.json").exists() else ""
        c = _load_json(rd / "critic_report.json") if (rd / "critic_report.json").exists() else {}
        k = _load_json(rd / "constraints_report.json") if (rd / "constraints_report.json").exists() else {}
        p = _load_json(rd / "analyst_proposal.json") if (rd / "analyst_proposal.json").exists() else {}

        rows.append({
            "run_dir": run_dir.name,
            "round": rd.name,
            "ask_perception_calls": conv.count("ask_perception:"),
            "perception_followups": conv.count("Perception follow-up"),
            "critic_status": c.get("status", ""),
            "critic_flags_count": len(c.get("critic_flags", []) or []),
            "constraints_count": k.get("count", 0),
            "proposal_changed_count": p.get("changed_count", 0),
            "has_critic_feedback": int((rd / "critic_feedback.json").exists()),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "round",
        "ask_perception_calls",
        "perception_followups",
        "critic_status",
        "critic_flags_count",
        "constraints_count",
        "proposal_changed_count",
        "has_critic_feedback",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved: {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a run directory containing roundN folders")
    ap.add_argument("--out", default=None, help="Output csv path (default: <run-dir>/phase2_evidence.csv)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "phase2_evidence.csv"
    export(run_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
