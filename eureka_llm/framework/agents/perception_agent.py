"""
perception_agent.py — Observes raw training data, produces structured perception report.

Role in the multi-agent system:
    Training → Perception Agent → perception_report → Analyst Agent

This agent does NOT write code or propose changes. It only observes and describes.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure framework directory is on path for imports
_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


PERCEPTION_SYSTEM_PROMPT = """You are a Perception Agent — an AI that observes raw training data
and produces concise, structured reports about agent behavior.

You do NOT propose changes. You do NOT write code. You only observe and describe."""


def build_perception_prompt(run_dir: Path, template_path: Path) -> str:
    """Build the perception prompt from training results and template."""
    # Dynamic path insert for direct imports (avoid relative import issues)
    _import_dir = Path(__file__).resolve().parent.parent
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))
    from template_engine import (
        load_training_data,
        format_metrics_table,
        format_env_metrics_section,
        format_component_table,
        format_traj_env_metrics_table,
        format_dynamics_section,
        format_tdrq_section,
    )

    data = load_training_data(run_dir)
    template = template_path.read_text(encoding="utf-8")

    traj = data["traj_summary"]
    lens = traj.get("lengths", {})

    placeholders = {
        "metrics_table": format_metrics_table(data["eval_history"]),
        "env_metrics_section": format_env_metrics_section(data["eval_history"]),
        "component_table": format_component_table(data["traj_summary"]),
        "traj_env_metrics_table": format_traj_env_metrics_table(data["traj_summary"]),
        "dynamics_section": format_dynamics_section(data["traj_summary"], run_dir),
        "tdrq_section": format_tdrq_section(data["traj_summary"], run_dir),
        "n_traj_episodes": str(traj.get("n_episodes", 0)),
        "traj_len_mean": str(lens.get("mean", "?")),
        "traj_len_min": str(lens.get("min", "?")),
        "traj_len_max": str(lens.get("max", "?")),
    }

    result = template
    for key, value in placeholders.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def extract_behavior_metrics(perception_report: str) -> dict:
    """Extract key numerical metrics from perception report for budget calculation.

    Only extracts mean_length (environment-agnostic) from perception report.
    Env-specific metrics come from env_metadata.
    """
    metrics = {}

    # Try to extract from "Key Numbers for Budget Calculation" section
    section_match = re.search(
        r"### 6\. Key Numbers.*?\n(.*?)(?=\n###|\Z)",
        perception_report, re.DOTALL
    )
    if section_match:
        section = section_match.group(1)
        for line in section.split("\n"):
            if "mean_length" in line.lower():
                nums = re.findall(r"[-+]?\d*\.?\d+", line)
                if nums:
                    metrics["mean_length"] = float(nums[0])
    return metrics


def run_perception_agent(run_dir: Path, api_key: str,
                          model: str = "deepseek-reasoner",
                          temperature: float = 0.3) -> str:
    """Run the perception agent on a completed training run.

    Args:
        run_dir: Training run directory (roundN/)
        api_key: LLM API key
        model: Model name
        temperature: Lower temperature for more factual output

    Returns:
        perception_report: Markdown report string
    """
    template_path = Path(__file__).resolve().parent.parent.parent / "templates" / "perception_prompt.txt"
    if not template_path.exists():
        return _generate_fallback_report(run_dir)

    prompt = build_perception_prompt(run_dir, template_path)
    report = call_llm(prompt, api_key, model, temperature)

    # Save artifacts
    (run_dir / "perception_prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "perception_response.md").write_text(report, encoding="utf-8")
    (run_dir / "perception_report.md").write_text(report, encoding="utf-8")

    return report


def _generate_fallback_report(run_dir: Path) -> str:
    """Generate a basic report without LLM call (if template is missing)."""
    import csv
    evals = []
    csv_path = run_dir / "evaluations" / "history.csv"
    if csv_path.exists():
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                evals.append(row)

    if not evals:
        return "No evaluation data available."

    last = evals[-1]
    report = (
        f"## Perception Report (Fallback)\n\n"
        f"### Evaluation Summary\n"
        f"- Mean length: {last.get('mean_length', 'N/A')}\n"
        f"- Env metrics: {last.get('env_metrics', 'N/A')}\n\n"
        f"Note: Full perception analysis requires LLM call with template."
    )
    (run_dir / "perception_report.md").write_text(report, encoding="utf-8")
    return report
