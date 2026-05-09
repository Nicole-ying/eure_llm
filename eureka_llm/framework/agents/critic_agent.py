"""critic_agent.py — Critic pass for proposal quality checks (Phase-2)."""

from pathlib import Path
import json


def run_critic_agent(round_dir: Path, proposal: dict, constraints_report: dict) -> dict:
    violations = constraints_report.get("violations", [])
    flagged = []
    if proposal.get("changed_count", 0) > 2:
        flagged.append("Proposal changes too many items; risk of unstable credit assignment.")
    for v in violations:
        if v.get("severity") == "high":
            flagged.append(f"High-severity principle violation present: {v.get('principle')}")
    out = {
        "status": "needs_revision" if flagged else "pass",
        "critic_flags": flagged,
        "recommendation": "Address high-severity violations first." if flagged else "Proposal acceptable for next iteration.",
    }
    (round_dir / "critic_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from memory.memory_system import MemorySystem
        mem = MemorySystem(round_dir.parent if round_dir.name.startswith("round") else round_dir)
        mem.update_belief("critic", {"round": round_dir.name, "status": out["status"], "flags": out["critic_flags"][:3]})
    except Exception:
        pass
    return out
