#!/usr/bin/env python3
"""Validate Phase-2 multi-agent completion evidence from a run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def _find_round_dirs(run_dir: Path) -> list[Path]:
    out = []
    for p in run_dir.glob("round*"):
        if p.is_dir() and re.fullmatch(r"round\d+", p.name):
            out.append(p)
    return sorted(out, key=lambda x: int(x.name.replace("round", "")))


def _read_text(path: Path) -> str:
    try:
        return path.read_text("utf-8")
    except Exception:
        return ""


def validate(run_dir: Path) -> dict:
    rounds = _find_round_dirs(run_dir)
    events_seen = set()
    ask_perception_hits = 0
    perception_followup_hits = 0
    belief_hits = set()
    critic_reports = 0
    constraints_reports = 0
    critic_feedback_reports = 0

    for rd in rounds:
        log = _read_text(rd / "experiment.log") + "\n" + _read_text(run_dir / "experiment.log")
        for ev in [
            "perception.completed",
            "analyst.started",
            "analyst.completed",
            "constraints.completed",
            "critic.completed",
        ]:
            if ev in log:
                events_seen.add(ev)

        conv_path = rd / "analyst_conversation.json"
        if conv_path.exists():
            txt = _read_text(conv_path)
            ask_perception_hits += txt.count("ask_perception:")
            perception_followup_hits += txt.count("Perception follow-up")

        if (rd / "critic_report.json").exists():
            critic_reports += 1
            try:
                payload = json.loads((rd / "critic_report.json").read_text("utf-8"))
                if payload.get("status") == "needs_revision" and (rd / "critic_feedback.json").exists():
                    critic_feedback_reports += 1
            except Exception:
                pass

        if (rd / "constraints_report.json").exists():
            constraints_reports += 1

    # memory scan
    mem_text = ""
    for p in [run_dir / "MEMORY.md", run_dir / "memory" / "MEMORY.md", run_dir / "memory" / "beliefs.json"]:
        if p.exists():
            mem_text += _read_text(p)
    for k in ["perception", "analyst", "generator", "critic", "constraints"]:
        if k in mem_text:
            belief_hits.add(k)

    checks = {
        "events_ok": len(events_seen) >= 4,
        "belief_ok": len(belief_hits) >= 4,
        "bidirectional_ok": ask_perception_hits > 0 and perception_followup_hits > 0,
        "critic_constraints_ok": critic_reports > 0 and constraints_reports > 0,
    }
    score = sum(1 for v in checks.values() if v)
    return {
        "run_dir": str(run_dir),
        "round_count": len(rounds),
        "events_seen": sorted(events_seen),
        "belief_agents_seen": sorted(belief_hits),
        "ask_perception_hits": ask_perception_hits,
        "perception_followup_hits": perception_followup_hits,
        "critic_reports": critic_reports,
        "constraints_reports": constraints_reports,
        "critic_feedback_reports": critic_feedback_reports,
        "checks": checks,
        "score": score,
        "status": "PASS" if score >= 3 else "NEEDS_WORK",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a run directory containing roundN folders")
    ap.add_argument("--out", default=None, help="Optional output json path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    result = validate(run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(args.out) if args.out else run_dir / "phase2_validation.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
