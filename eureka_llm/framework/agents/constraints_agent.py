"""constraints_agent.py — Non-LLM constraints manager for Phase-2."""

from pathlib import Path
import sys

_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from template_engine import load_training_data
from constraint_discovery import detect_constraint_violations


def run_constraints_agent(run_dir: Path) -> dict:
    data = load_training_data(run_dir)
    violations = detect_constraint_violations(data.get("traj_summary", {}), data.get("eval_history", []))
    out = {"violations": violations, "count": len(violations)}
    (run_dir / "constraints_report.json").write_text(__import__("json").dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from memory.memory_system import MemorySystem
        mem = MemorySystem(run_dir.parent if run_dir.name.startswith("round") else run_dir)
        mem.update_belief("constraints", {"round": run_dir.name, "count": out["count"]})
    except Exception:
        pass
    return out
