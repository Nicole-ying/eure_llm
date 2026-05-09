#!/usr/bin/env python3
"""Aggregate *_prompt_compaction.json across rounds into a single CSV."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="Root directory containing run folders")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    rows = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        for round_dir in sorted(run_dir.glob("round*")):
            for fname in ("perception_prompt_compaction.json", "analyst_prompt_compaction.json", "generator_prompt_compaction.json"):
                p = round_dir / fname
                if not p.exists():
                    continue
                payload = json.loads(p.read_text("utf-8"))
                for section, st in payload.items():
                    if not isinstance(st, dict):
                        continue
                    src = int(st.get("source_lines", 0))
                    kept = int(st.get("kept_lines", 0))
                    dropped = int(st.get("dropped_lines", max(0, src - kept)))
                    keep_ratio = (kept / src) if src > 0 else 0.0
                    rows.append({
                        "run": run_dir.name,
                        "round": round_dir.name,
                        "file": fname,
                        "section": section,
                        "source_lines": src,
                        "kept_lines": kept,
                        "dropped_lines": dropped,
                        "keep_ratio": f"{keep_ratio:.4f}",
                    })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "round", "file", "section", "source_lines", "kept_lines", "dropped_lines", "keep_ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
