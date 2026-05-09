"""Utilities for configurable prompt compaction and retention statistics."""

from __future__ import annotations
import json
from pathlib import Path


def load_prompt_policy(run_dir: Path, agent: str) -> dict:
    cfg_path = run_dir / "config.yaml"
    default = {
        "max_lines_markdown": 80,
        "max_lines_memory": 60,
        "max_env_metrics_table": 6,
        "max_env_metrics_section": 6,
    }
    if not cfg_path.exists():
        return default
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text("utf-8")) or {}
        p = (((cfg.get("phase3") or {}).get("prompt_policy") or {}).get(agent) or {})
        out = dict(default)
        out.update({k: v for k, v in p.items() if isinstance(v, int)})
        return out
    except Exception:
        return default


def summarize_structured_lines(text: str, max_lines: int, keywords: tuple[str, ...] = ()) -> tuple[str, dict]:
    if not text:
        return text, {"source_lines": 0, "kept_lines": 0, "dropped_lines": 0}
    lines = [ln for ln in text.splitlines() if ln.strip()]
    keep = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(("#", "###", "-", "*", "|")) or any(k in s.lower() for k in keywords):
            keep.append(ln)
    if not keep:
        keep = lines
    kept = keep[:max_lines]
    return "\n".join(kept), {
        "source_lines": len(lines),
        "candidate_lines": len(keep),
        "kept_lines": len(kept),
        "dropped_lines": max(0, len(lines) - len(kept)),
    }


def write_compaction_stats(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
