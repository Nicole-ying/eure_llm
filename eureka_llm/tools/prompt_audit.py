#!/usr/bin/env python3
"""Audit prompt templates for zero-shot compliance and produce a markdown report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
OUT = ROOT / "docs" / "prompt_zero_shot_audit.md"
OUT_JSON = ROOT / "docs" / "prompt_zero_shot_audit.json"


@dataclass
class Finding:
    level: str
    rule: str
    line: int
    text: str


def scan_template(path: Path) -> list[Finding]:
    lines = path.read_text(encoding="utf-8").splitlines()
    findings: list[Finding] = []
    banned_env = re.compile(r"\b(MountainCar|LunarLander|BipedalWalker|HalfCheetah|Ant|Humanoid|Walker2d)\b", re.I)
    absolute_thr = re.compile(r"\b(>\s*\d+(\.\d+)?|<\s*\d+(\.\d+)?|exactly\s+\d+(\.\d+)?)\b", re.I)
    # Avoid over-flagging generic instructional wording like "should be simple".
    # Only flag direct answer-leak language.
    answer_hint = re.compile(
        r"\b(correct answer|ground truth answer|must be this answer|the answer is)\b",
        re.I,
    )
    principle_keywords = ("principle", "alignment", "consistency", "balance", "relative")
    has_principle_scaffold = any(any(k in s.lower() for k in principle_keywords) for s in lines)

    for i, s in enumerate(lines, start=1):
        if banned_env.search(s):
            findings.append(Finding("fail", "environment_specific_reference", i, s.strip()))
        if absolute_thr.search(s):
            findings.append(Finding("warn", "absolute_threshold_language", i, s.strip()))
        if answer_hint.search(s):
            findings.append(Finding("fail", "embedded_correct_answer_hint", i, s.strip()))

    if not has_principle_scaffold:
        findings.append(Finding("warn", "missing_principle_or_relative_language", 1, "No principle/relative scaffold detected"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit prompt templates for zero-shot compliance.")
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return non-zero if fail findings exist (for CI/policy gate).",
    )
    parser.add_argument(
        "--apply-fixes",
        action="store_true",
        help="Apply safe text-level fixes for known warning patterns before auditing.",
    )
    args = parser.parse_args()

    files = sorted(TEMPLATES.glob("*prompt*.txt"))
    lines = ["# Prompt Zero-Shot Audit", ""]
    total_fail = 0
    total_warn = 0
    applied_fixes = []
    for p in files:
        if args.apply_fixes:
            original = p.read_text(encoding="utf-8")
            fixed = original.replace("exactly 0", "near-zero")
            if fixed != original:
                p.write_text(fixed, encoding="utf-8")
                applied_fixes.append({"file": str(p.relative_to(ROOT)), "fix": "replace 'exactly 0' -> 'near-zero'"})

        findings = scan_template(p)
        fails = [f for f in findings if f.level == "fail"]
        warns = [f for f in findings if f.level == "warn"]
        total_fail += len(fails)
        total_warn += len(warns)
        lines.append(f"## {p.name}")
        lines.append(f"- status: {'FAIL' if fails else 'PASS'}")
        lines.append(f"- fail_count: {len(fails)}")
        lines.append(f"- warn_count: {len(warns)}")
        if findings:
            lines.append("")
            lines.append("| level | rule | line | snippet |")
            lines.append("|---|---|---:|---|")
            for f in findings:
                snippet = f.text.replace("|", "\\|")
                lines.append(f"| {f.level} | {f.rule} | {f.line} | {snippet} |")
        lines.append("")

    lines.append("## Summary")
    lines.append(f"- total_fail: {total_fail}")
    lines.append(f"- total_warn: {total_warn}")
    if applied_fixes:
        lines.append(f"- applied_fixes: {len(applied_fixes)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "total_fail": total_fail,
                "total_warn": total_warn,
                "applied_fixes": applied_fixes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Audit report written to: {OUT}")
    print(f"Audit json written to: {OUT_JSON}")
    return 1 if (args.strict_exit and total_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
