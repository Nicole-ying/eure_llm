"""
memory_system.py — Persistent cross-round memory for the multi-agent reward framework.

Three-layer design (inspired by Claude Code's memory architecture):

Layer 1 — TASK_MANIFEST.md: Permanent task description. Created after round0,
         never modified. Every agent gets this in context every round.

Layer 2 — MEMORY.md: Cross-round lessons index. Max 200 lines. Each entry is
         a causal lesson: "what changed → what happened → why". Updated each round.

Layer 3 — Per-round storage: Full artifacts for each round (reward functions,
         training summaries, proposals, reflections). Accessed on-demand via
         query_memory().

Usage:
    mem = MemorySystem(run_dir)
    mem.initialize_task_manifest(step_source, task_description)
    mem.store_round(round_num, artifacts)
    mem.add_lesson("Round 2: overconstrained reward → agent stopped moving")
    lessons = mem.query_lessons("overconstrain")
    summary = mem.get_training_summary(2)
"""

import json
import re
import shutil
from pathlib import Path
from typing import Optional


MEMORY_HEADER = """# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

"""

# ── Phase-2 v2 Belief Schemas ─────────────────────────────────────────────
# Structured per-agent belief states with cross-round accumulation.

PERCEPTION_BELIEF_V2 = {
    "agent": "perception", "version": 2,
    "history": [],
    "trends": {},
    "dynamics_profile": "",
    "cross_round_insights": "",
}

ANALYST_BELIEF_V2 = {
    "agent": "analyst", "version": 2,
    "history": [],
    "validated_hypotheses": [],
    "causal_model": "",
}

GENERATOR_BELIEF_V2 = {
    "agent": "generator", "version": 2,
    "history": [],
    "code_patterns_worked": [],
    "code_patterns_failed": [],
}

CRITIC_BELIEF_V2 = {
    "agent": "critic", "version": 2,
    "history": [],
    "recurring_issues": [],
}

CONSTRAINTS_BELIEF_V2 = {
    "agent": "constraints", "version": 2,
    "history": [],
    "violation_history": {},
}

BELIEF_V2_SCHEMAS = {
    "perception": PERCEPTION_BELIEF_V2,
    "analyst": ANALYST_BELIEF_V2,
    "generator": GENERATOR_BELIEF_V2,
    "critic": CRITIC_BELIEF_V2,
    "constraints": CONSTRAINTS_BELIEF_V2,
}


# ── Phase-2 v2: helper functions ──────────────────────────────────────────

def _compute_perception_trends(history: list) -> dict:
    """Aggregate key metrics across history entries to detect cross-round trends."""
    if not history:
        return {}
    trends = {}
    # Track mean_length progression
    lengths = []
    for h in history:
        km = h.get("key_metrics", {}) or {}
        if "mean_length" in km:
            lengths.append(km["mean_length"])
    if lengths:
        trends["mean_length_trend"] = {
            "start": lengths[0], "end": lengths[-1],
            "direction": "increasing" if lengths[-1] > lengths[0] * 1.1
            else "decreasing" if lengths[-1] < lengths[0] * 0.9
            else "stable",
            "values": lengths,
        }
    # Count pattern frequency
    all_patterns = {}
    for h in history:
        for p in h.get("identified_patterns", []):
            all_patterns[p] = all_patterns.get(p, 0) + 1
    if all_patterns:
        trends["recurring_patterns"] = sorted(
            all_patterns.items(), key=lambda x: -x[1])[:5]
    return trends


def _summarize_perception_insights(history: list) -> str:
    """Generate a one-line cross-round insight from perception history."""
    if not history:
        return ""
    trends = _compute_perception_trends(history)
    parts = []
    length_t = trends.get("mean_length_trend", {})
    if length_t:
        parts.append(
            f"Episode length {length_t['direction']} "
            f"({length_t['start']:.0f} → {length_t['end']:.0f})")
    patterns = trends.get("recurring_patterns", [])
    if patterns:
        top = patterns[0]
        parts.append(f"Most frequent pattern: '{top[0]}' (x{top[1]})")
    return "; ".join(parts) if parts else ""


def _classify_outcome(outcome_text: str) -> str:
    """Classify an outcome text as confirmed/refuted/inconclusive."""
    lower = outcome_text.lower()
    improved_words = ["increas", "improv", "higher", "better", "success", "上升", "提高", "改善"]
    degraded_words = ["decreas", "degrad", "lower", "worse", "fail", "下降", "恶化", "失败"]
    if any(w in lower for w in improved_words):
        return "confirmed"
    if any(w in lower for w in degraded_words):
        return "refuted"
    return "inconclusive"


class RoundMemory:
    """Per-round artifact storage and retrieval."""

    def __init__(self, round_dir: Path):
        self.dir = round_dir

    @property
    def reward_fn_source(self) -> Optional[str]:
        p = self.dir / "reward_fn_source.py"
        return p.read_text("utf-8") if p.exists() else None

    @property
    def training_summary(self) -> Optional[dict]:
        """Load round-level training summary from evaluations/history.csv + trajectory_logs."""
        import csv
        evals = []
        csv_path = self.dir / "evaluations" / "history.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["timesteps"] = int(row["timesteps"]) if row.get("timesteps") else 0
                    raw = row.get("env_metrics", "{}")
                    try:
                        row["env_metrics"] = json.loads(raw)
                    except json.JSONDecodeError:
                        row["env_metrics"] = {}
                    evals.append(row)
        return {"eval_history": evals} if evals else None

    @property
    def perception_report(self) -> Optional[str]:
        p = self.dir / "perception_report.md"
        return p.read_text("utf-8") if p.exists() else None

    @property
    def analyst_proposal(self) -> Optional[dict]:
        p = self.dir / "analyst_proposal.json"
        return json.loads(p.read_text("utf-8")) if p.exists() else None

    @property
    def reflection(self) -> Optional[str]:
        p = self.dir / "reflection.md"
        return p.read_text("utf-8") if p.exists() else None

    @property
    def gif_path(self) -> Optional[Path]:
        gif_dir = self.dir / "gifs"
        if gif_dir.exists():
            gifs = sorted(gif_dir.glob("*.gif"))
            if gifs:
                return gifs[-1]
        return None


class MemorySystem:
    """Cross-round memory management."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.memory_dir = self.run_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.belief_dir = self.memory_dir / "beliefs"
        self.belief_dir.mkdir(parents=True, exist_ok=True)

    # ── Layer 1: Task Manifest ──

    @property
    def task_manifest_path(self) -> Path:
        return self.memory_dir / "TASK_MANIFEST.md"

    def get_task_manifest(self) -> str:
        p = self.task_manifest_path
        return p.read_text("utf-8") if p.exists() else ""

    def initialize_task_manifest(self, step_source: str,
                                  env_description: str = "",
                                  termination_analysis: str = "",
                                  obs_description: str = "",
                                  action_description: str = "") -> str:
        """Create the permanent task manifest from round0 context."""
        content = f"""# Task Manifest

## Environment Description
{env_description or "Inferred from step() source."}

## Termination Analysis
{termination_analysis or "See step() source for termination conditions."}

## Observation Space
{obs_description or "See step() source for observation structure."}

## Action Space
{action_description or "See step() source for action structure."}

## Step Source Code
```python
{step_source}
```
"""
        self.task_manifest_path.write_text(content, encoding="utf-8")
        return content

    # ── Layer 2: MEMORY.md (lessons index) ──

    @property
    def memory_md_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def get_lessons(self, max_lines: int = 200) -> str:
        """Get MEMORY.md content, truncated to max_lines."""
        p = self.memory_md_path
        if not p.exists():
            return ""
        lines = p.read_text("utf-8").splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"\n[TRUNCATED at {max_lines} lines]"]
        return "\n".join(lines)

    def add_lesson(self, lesson_text: str) -> None:
        """Append a lesson to MEMORY.md."""
        p = self.memory_md_path
        header = MEMORY_HEADER if not p.exists() else ""
        with p.open("a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(lesson_text.rstrip() + "\n\n")

    def query_lessons(self, keyword: str, max_results: int = 5) -> list[str]:
        """Search MEMORY.md for lessons relevant to keyword."""
        p = self.memory_md_path
        if not p.exists():
            return []
        text = p.read_text("utf-8")
        # Split by lesson entries (double-newline separated)
        lessons = re.split(r"\n\n+", text)
        matches = []
        for lesson in lessons:
            lesson = lesson.strip()
            if not lesson or lesson.startswith("#"):
                continue
            if keyword.lower() in lesson.lower():
                matches.append(lesson)
        return matches[:max_results]

    # ── Layer 3: Per-round storage ──

    def round_path(self, round_num: int) -> Path:
        return self.run_dir / f"round{round_num}"

    def get_round(self, round_num: int) -> RoundMemory:
        return RoundMemory(self.round_path(round_num))

    def get_available_rounds(self) -> list[int]:
        """Return sorted list of round numbers that exist."""
        rounds = []
        for d in self.run_dir.iterdir():
            if d.name.startswith("round") and d.name[5:].isdigit():
                rounds.append(int(d.name[5:]))
        return sorted(rounds)

    def get_recent_lessons(self, n: int = 3) -> str:
        """Get the n most recent complete rounds as a summarized history string."""
        rounds = self.get_available_rounds()
        recent = rounds[-n:] if len(rounds) > n else rounds
        parts = []
        for r in recent:
            rm = self.get_round(r)
            summary_parts = [f"### Round {r}"]
            # Reward function
            rsrc = rm.reward_fn_source
            if rsrc:
                # Extract just the docstring/comments, not full code
                lines = rsrc.splitlines()
                doc_lines = [l for l in lines if l.strip().startswith(("#", '"""'))][:5]
                summary_parts.append("Reward: " + " ".join(doc_lines)[:200])

            # Training summary
            ts = rm.training_summary
            if ts and ts.get("eval_history"):
                last = ts["eval_history"][-1]
                summary_parts.append(
                    f"Metrics: completion={last.get('completion_rate', '?')}, "
                    f"fall={last.get('fall_rate', '?')}, "
                    f"mean_len={last.get('mean_length', '?')}"
                )

            # Perception report excerpt
            pr = rm.perception_report
            if pr:
                lines = pr.splitlines()
                behavior_lines = [l for l in lines if "behavior" in l.lower()
                                  or "trend" in l.lower() or "summary" in l.lower()][:3]
                if behavior_lines:
                    summary_parts.append("Perception: " + " ".join(behavior_lines)[:200])

            # Reflection
            ref = rm.reflection
            if ref:
                ref_lines = ref.splitlines()[:3]
                summary_parts.append("Lesson: " + " ".join(ref_lines)[:200])

            parts.append("\n".join(summary_parts))
        return "\n\n".join(parts)

    # ── Reward Budget Calculator ──

    def calculate_reward_budget(self, reward_code: str,
                                 behavior_metrics: dict,
                                 component_means: dict[str, float]) -> dict:
        """
        Estimate whether proposed reward function changes would produce
        positive net reward for the current behavior policy.

        Generic version — uses trajectory component_means if available,
        otherwise falls back to heuristic from reward code parsing.

        Returns:
            dict with component estimates and total estimate
        """
        estimates = {}
        mean_len = behavior_metrics.get("mean_length", 1000.0)
        fall_rate = behavior_metrics.get("fall_rate", 0.5)

        # Use trajectory component_means if available (most reliable)
        if component_means:
            for comp, mean_val in component_means.items():
                if isinstance(mean_val, (int, float)):
                    estimates[comp] = round(float(mean_val), 4)

        # Find termination penalties in reward code (any env)
        term_patterns = [
            r'if\s+terminated.*?return\s*([-]?\d+\.?\d*)',
            r'r_term\s*=\s*([-]?\d+\.?\d*)',
            r'termination.*?=\s*([-]?\d+\.?\d*)',
        ]
        term_w = 50.0
        for pat in term_patterns:
            m = re.search(pat, reward_code)
            if m:
                term_w = abs(float(m.group(1)))
                break

        amortized_term = -(term_w * fall_rate) / max(mean_len, 1)
        estimated_total = sum(estimates.values()) + amortized_term

        return {
            "component_estimates": estimates,
            "termination_amortized_per_step": round(amortized_term, 4),
            "estimated_total_per_step": round(estimated_total, 4),
            "is_positive": estimated_total > 0,
            "warning": "Net reward near zero or negative — agent may stop moving"
            if estimated_total < 0.05 else None,
            "based_on_behavior": dict(behavior_metrics),
        }

    # ── Phase-2: persistent per-agent belief states ──
    def belief_path(self, agent_name: str) -> Path:
        return self.belief_dir / f"{agent_name}.json"

    def get_belief(self, agent_name: str) -> dict:
        p = self.belief_path(agent_name)
        if not p.exists():
            return {"agent": agent_name, "version": 1, "history": []}
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return {"agent": agent_name, "version": 1, "history": []}

    def update_belief(self, agent_name: str, entry: dict, max_entries: int = 50) -> dict:
        belief = self.get_belief(agent_name)
        belief.setdefault("agent", agent_name)
        belief.setdefault("version", 1)
        hist = belief.setdefault("history", [])
        hist.append(entry)
        if len(hist) > max_entries:
            belief["history"] = hist[-max_entries:]
        self.belief_path(agent_name).write_text(json.dumps(belief, ensure_ascii=False, indent=2), encoding="utf-8")
        return belief

    # ── Phase-2 v2: Migration ──────────────────────────────────────────

    def _migrate_v1_to_v2(self, belief: dict, expected_agent: str) -> dict:
        """Upgrade v1 flat-history belief to v2 structured schema (idempotent)."""
        if belief.get("version", 1) >= 2:
            return belief
        base = dict(BELIEF_V2_SCHEMAS.get(expected_agent, {
            "agent": expected_agent, "version": 2, "history": [],
        }))
        base["history"] = belief.get("history", [])
        return base

    # ── Phase-2 v2: Cross-agent belief reading ─────────────────────────

    def get_agent_beliefs(self, agent_names: list[str]) -> dict[str, dict]:
        """Read and auto-migrate beliefs for multiple agents at once.

        Returns dict keyed by agent name, each value is a v2-migrated belief dict.
        Useful for injecting cross-agent context into prompts.
        """
        result = {}
        for name in agent_names:
            raw = self.get_belief(name)
            result[name] = self._migrate_v1_to_v2(raw, name)
        return result

    def format_beliefs_for_prompt(self, agent_names: list[str],
                                   max_history: int = 3) -> str:
        """Format selected agent beliefs as a compact markdown snippet for prompt injection."""
        beliefs = self.get_agent_beliefs(agent_names)
        parts = []
        for name, b in beliefs.items():
            history = b.get("history", [])[-max_history:]
            if not history:
                continue
            lines = [f"### {name.title()} Agent Belief State"]
            for h in history:
                lines.append(f"- Round {h.get('round', h.get('round_num', '?'))}: "
                             f"{str(h.get('diagnosis', h.get('summary', h.get('status', ''))))[:120]}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ── Phase-2 v2: Structured Perception belief update ────────────────

    def update_perception_belief_v2(self, round_name: str,
                                     behavior_summary: str = "",
                                     key_metrics: dict = None,
                                     dynamics_trend: str = "",
                                     identified_patterns: list = None,
                                     anomalies: list = None) -> dict:
        """Update perception belief with structured observation record (v2 schema)."""
        belief = self._migrate_v1_to_v2(
            self.get_belief("perception"), "perception")
        entry = {
            "round": round_name,
            "behavior_summary": behavior_summary[:300],
            "key_metrics": key_metrics or {},
            "dynamics_trend": dynamics_trend,
            "identified_patterns": (identified_patterns or [])[:5],
            "anomalies": (anomalies or [])[:3],
        }
        hist = belief.setdefault("history", [])
        hist.append(entry)
        if len(hist) > 50:
            belief["history"] = hist[-50:]

        # Recompute cross-round trends (simplified: aggregate key metrics across rounds)
        belief["trends"] = _compute_perception_trends(belief["history"])
        belief["cross_round_insights"] = _summarize_perception_insights(belief["history"])
        belief["version"] = 2

        self.belief_path("perception").write_text(
            json.dumps(belief, ensure_ascii=False, indent=2), encoding="utf-8")
        return belief

    # ── Phase-2 v2: Structured Analyst belief update ───────────────────

    def update_analyst_belief_v2(self, round_num: int, diagnosis: str = "",
                                  violated_principle: str = "",
                                  root_cause_category: str = "",
                                  changed_count: int = 0,
                                  proposed_changes_summary: str = "",
                                  predicted_effect: str = "",
                                  actual_outcome: str = None) -> dict:
        """Update analyst belief with structured proposal record (v2 schema)."""
        belief = self._migrate_v1_to_v2(
            self.get_belief("analyst"), "analyst")
        entry = {
            "round": round_num,
            "diagnosis": diagnosis[:300],
            "violated_principle": violated_principle,
            "root_cause_category": root_cause_category,
            "changed_count": changed_count,
            "proposed_changes_summary": proposed_changes_summary[:200],
            "predicted_effect": predicted_effect[:200],
            "actual_outcome": actual_outcome,
        }
        hist = belief.setdefault("history", [])
        hist.append(entry)
        if len(hist) > 50:
            belief["history"] = hist[-50:]

        # Track validated hypotheses
        if actual_outcome and len(hist) >= 2:
            prev = hist[-2]
            if prev.get("actual_outcome") is None and prev.get("predicted_effect"):
                hypothesis = {
                    "round_proposed": prev["round"],
                    "hypothesis": prev["predicted_effect"][:150],
                    "outcome_round": round_num,
                    "outcome": _classify_outcome(actual_outcome),
                }
                belief.setdefault("validated_hypotheses", []).append(hypothesis)

        belief["version"] = 2
        self.belief_path("analyst").write_text(
            json.dumps(belief, ensure_ascii=False, indent=2), encoding="utf-8")
        return belief
