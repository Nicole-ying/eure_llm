"""
pipeline.py — Multi-agent reward function iteration pipeline.

Orchestrates the full agent workflow:
    Round 0: LLM generates initial reward → Train → Perception → Task Manifest
    Round N: Perception → Analyst (ReAct) → Generator → Train → Reflection → Memory

Usage:
    python pipeline.py --mode round0 --env-dir envs/BipedalWalker-v3/ ...
    python pipeline.py --mode iterate --run-dir runs/my_experiment/ --round 1 ...

This preserves the EXACT same experiment logging structure as the original framework.
All training (train.py), evaluation (evaluate.py), and behavior metric collection
logic is unchanged.
"""

import argparse
import json
import os
import re
import shutil
import sys
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter
from typing import Callable

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Ensure framework directory is on path for imports
_framework_dir = Path(__file__).resolve().parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from runtime_policy import compute_evidence_fingerprint, should_rerun_analyst

from llm_call import call_llm, extract_reward_fn, save_artifacts
from template_engine import (
    build_round0_prompt,
    derive_reward_constraints,
    load_training_data,
)
from memory.memory_system import MemorySystem


BEIJING = timezone(timedelta(hours=8))


def _safe_load_config_text(text: str) -> dict:
    if yaml is not None:
        return yaml.safe_load(text) or {}
    # Minimal fallback parser for dry-run when PyYAML is unavailable.
    # Supports only top-level "key: value" pairs.
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def _safe_dump_config(cfg: dict) -> str:
    if yaml is not None:
        return yaml.safe_dump(cfg, sort_keys=False)
    import json
    return json.dumps(cfg, ensure_ascii=False, indent=2)


class _LogFile:
    """Shared log file handle for both stdout and stderr tee."""
    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, data: str):
        self.handle.write(data)

    def flush(self):
        self.handle.flush()


class _Tee:
    """Tee a stream to both console and the shared log file."""
    def __init__(self, stream, log_file: _LogFile):
        self.stream = stream
        self.log = log_file

    def write(self, data: str):
        self.stream.write(data)
        self.log.write(data)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def _setup_logging(exp_dir: Path) -> _Tee:
    """Redirect stdout and stderr to tee into experiment.log."""
    log_file = _LogFile(exp_dir / "experiment.log")
    stdout_tee = _Tee(sys.stdout, log_file)
    stderr_tee = _Tee(sys.stderr, log_file)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    return stdout_tee


_ENV_DESC_DIR = Path(__file__).resolve().parent.parent / "env_descriptions"


def _load_env_description(env_name: str) -> str:
    """Load environment description markdown for a given env name.

    Args:
        env_name: Environment directory name, e.g. "HalfCheetah-v4"

    Returns:
        Description string, or empty string if no description file exists.
    """
    desc_path = _ENV_DESC_DIR / f"{env_name}.md"
    if desc_path.exists():
        return desc_path.read_text("utf-8").strip()
    return ""


def _run_subprocess(cmd: list) -> subprocess.CompletedProcess:
    """Run subprocess, piping stdout/stderr through the Tee logger in real time.

    Captures stdout and stderr separately for self-heal error analysis.
    """
    import threading
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    stdout_lines = []
    stderr_lines = []

    def _read(stream, collector):
        for line in iter(stream.readline, ""):
            print(line, end="")
            collector.append(line)
        stream.close()

    t1 = threading.Thread(target=_read, args=(proc.stdout, stdout_lines))
    t2 = threading.Thread(target=_read, args=(proc.stderr, stderr_lines))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    proc.wait()

    return subprocess.CompletedProcess(
        proc.args, proc.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


def _write_prompt_efficiency_report(round_dir: Path):
    """Create a markdown report from prompt compaction and guard artifacts."""
    items = [
        ("perception_prompt_compaction.json", "Perception Compaction"),
        ("analyst_prompt_compaction.json", "Analyst Compaction"),
        ("generator_prompt_compaction.json", "Generator Compaction"),
        ("perception_guard.json", "Perception Guard"),
        ("analyst_guard.json", "Analyst Guard"),
        ("reflection_guard.json", "Reflection Guard"),
    ]
    lines = ["# Prompt Efficiency Report", ""]
    for fname, title in items:
        p = round_dir / fname
        if not p.exists():
            continue
        lines.append(f"## {title}")
        try:
            payload = json.loads(p.read_text("utf-8"))
        except Exception:
            lines.append("- Failed to parse JSON.")
            lines.append("")
            continue
        if "passed" in payload:
            lines.append(f"- passed: {payload.get('passed')}")
            lines.append(f"- env_leakage: {payload.get('env_leakage')}")
            lines.append(f"- implicit_env_hint: {payload.get('implicit_env_hint')}")
            lines.append(f"- absolute_threshold_language: {payload.get('absolute_threshold_language')}")
        else:
            lines.append("| Section | Source | Kept | Dropped | Keep Ratio |")
            lines.append("|---|---:|---:|---:|---:|")
            for sec, st in payload.items():
                if not isinstance(st, dict):
                    continue
                src = st.get("source_lines", 0)
                kept = st.get("kept_lines", 0)
                drop = st.get("dropped_lines", max(0, src - kept))
                ratio = (kept / src) if src else 0.0
                lines.append(f"| {sec} | {src} | {kept} | {drop} | {ratio:.3f} |")
        lines.append("")
    (round_dir / "prompt_efficiency_report.md").write_text("\n".join(lines), encoding="utf-8")


@dataclass
class Event:
    """A named event with payload, source, and timestamp."""
    name: str
    payload: dict = field(default_factory=dict)
    source: str = ""
    timestamp: float = 0.0


class EventCoordinator:
    """Event-driven coordinator for multi-agent orchestration (Phase-2 step 1).

    Agents communicate via named events instead of hardcoded function calls.
    The coordinator maintains an event log, shared context, and subscriber list.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._event_log: list[Event] = []
        self._context: dict = {}

    def on(self, event_name: str, handler: Callable):
        """Subscribe to an event. handler receives the Event object."""
        self._subscribers.setdefault(event_name, []).append(handler)

    def emit(self, event_name: str, payload: dict, source: str = ""):
        """Publish an event, triggering all subscribed handlers synchronously."""
        event = Event(
            name=event_name, payload=payload,
            source=source, timestamp=time.time(),
        )
        self._event_log.append(event)
        # Print event summary for console visibility
        summary = str(payload)[:80]
        print(f"  [event] {event_name} {summary}")
        for handler in self._subscribers.get(event_name, []):
            handler(event)

    def get_event_log(self, last_n: int = 20) -> list[dict]:
        """Return recent event summaries (for audit / prompt injection)."""
        return [
            {"event": e.name, "source": e.source,
             "summary": str(e.payload)[:120]}
            for e in self._event_log[-last_n:]
        ]

    @property
    def context(self) -> dict:
        """Shared mutable context accessible by all handlers.

        Used to pass state (proposal, code, reports) between event handlers.
        """
        return self._context


# ─────────────────────────────────────────────────────────────────────────────
# Round 0
# ─────────────────────────────────────────────────────────────────────────────

def run_round0(env_dir: Path, exploration_path: Path, config_path: Path,
               output_dir: Path, api_key: str, model: str = "deepseek-reasoner",
               temperature: float = 0.6, dry_run: bool = False,
               task_description: str = "") -> dict:
    """Round 0: generate initial reward function from environment exploration.

    This is the same as the original round0 flow, but additionally creates
    the task manifest for future rounds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build round0 prompt (same as original)
    template_path = Path(__file__).resolve().parent.parent / "templates" / "round0_prompt.txt"
    prompt = build_round0_prompt(env_dir, template_path, exploration_path,
                                 task_description=task_description)

    if dry_run:
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"[dry-run] Round 0 prompt → {output_dir / 'prompt.txt'}")
        return {"code": "# dry-run"}

    # Call LLM
    print(f"\n{'='*60}")
    print("  Round 0: Generating initial reward function")
    print(f"{'='*60}")
    response = call_llm(prompt, api_key, model, temperature)

    # Extract and save code
    code = extract_reward_fn(response)
    save_artifacts(output_dir, prompt, response, code=code)

    # Save reward source
    header = '"""LLM-generated reward function.\nSource: round0\n"""\n\nimport math\nimport numpy as np\n\n'
    reward_path = output_dir / "reward_fn_source.py"
    reward_path.write_text(header + code + "\n", encoding="utf-8")
    print(f"  Reward source → {reward_path}")

    return {
        "code": code,
        "prompt": prompt,
        "response": response,
        "output_dir": output_dir,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent iteration (round 1+)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_missing_reflection(round_dir: Path, round_num: int,
                               memory_system: MemorySystem,
                               api_key: str, model: str,
                               temperature: float) -> bool:
    """Patch a round that completed training but lacks reflection.

    Checks if round{round_num} has evaluation data but no reflection.md.
    If missing, runs perception agent + reflection agent to fill the gap.
    Called before starting new rounds in --continue mode.

    Returns True if the round is now fully reflected (or was already).
    """
    if not round_dir.exists():
        return False

    # Check if training completed (has eval history with data)
    eval_path = round_dir / "evaluations" / "history.csv"
    if not eval_path.exists():
        return False
    with eval_path.open("r") as f:
        import csv
        has_data = any(row for row in csv.DictReader(f))
    if not has_data:
        return False

    # Ensure perception report exists
    perception_path = round_dir / "perception_report.md"
    if not perception_path.exists():
        print(f"  [resume] Round {round_num}: perception missing, generating...")
        from agents.perception_agent import run_perception_agent
        run_perception_agent(round_dir, api_key, model, temperature=0.3)
        if not perception_path.exists():
            print(f"  [resume] Round {round_num}: perception agent failed.")
            return False

    # Ensure reflection exists
    reflection_path = round_dir / "reflection.md"
    if not reflection_path.exists():
        print(f"  [resume] Round {round_num}: reflection missing, generating...")
        from agents.reflection_agent import run_reflection_agent
        run_reflection_agent(round_dir, round_num, memory_system,
                             api_key, model, temperature=0.3)
        if not reflection_path.exists():
            print(f"  [resume] Round {round_num}: reflection agent failed.")
            return False
        print(f"  [resume] Round {round_num}: MEMORY.md updated.")

    return True


def run_iteration(run_dir: Path, env_dir: Path, round_num: int,
                   exploration_path: Path, config: dict,
                   api_key: str, model: str = "deepseek-reasoner",
                   temperature: float = 0.6,
                   dry_run: bool = False,
                   skip_train: bool = False) -> dict:
    """Run one multi-agent iteration round using event-driven orchestration.

    Full agent workflow (event-driven):
        1. Perception Agent → perception_report.md
        2. Analyst Agent (ReAct) → analyst_proposal.json
        3. Constraints + Critic Agents (mid/late rounds only)
        4. Generator Agent → reward_fn_source.py
        5. Train → new policy
        6. Perception Agent (analyze this round)
        7. Reflection Agent → reflection.md + MEMORY.md update

    Returns:
        Dict with iteration results (same contract as original).
    """
    prev_round_dir = run_dir / f"round{round_num - 1}"
    output_dir = run_dir / f"round{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_system = MemorySystem(run_dir)
    coordinator = EventCoordinator()

    # Dynamic role policy
    rp = (config.get("phase2", {}) or {}).get("role_policy", {})
    early_max = int(rp.get("early_max_round", 2))
    mid_max = int(rp.get("mid_max_round", 5))
    role_stage = "early" if round_num <= early_max else ("mid" if round_num <= mid_max else "late")

    # Shared context for handler communication
    ctx = coordinator.context
    ctx.update({
        "round_num": round_num,
        "role_stage": role_stage,
        "prev_round_dir": prev_round_dir,
        "output_dir": output_dir,
        "memory_system": memory_system,
        "config": config,
        "run_dir": run_dir,
        "env_dir": env_dir,
        "exploration_path": exploration_path,
        # Mutable step results (handlers set these)
        "proposal": {"diagnosis": "not_run", "changed_count": 0, "proposed_changes": []},
        "constraints_report": {"violations": [], "count": 0},
        "critic_report": {"status": "pass", "critic_flags": []},
        "code": None,
        "reward_path": None,
        "result": None,
    })

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Iteration — Round {round_num}")
    print(f"  Run dir: {run_dir.name}")
    print(f"  Previous: round{round_num - 1}")
    print(f"  Role stage: {role_stage}")
    print(f"{'='*60}")

    # ─── Step 1: Perception on previous round ───────────────────────────
    def _on_iteration_start(event: Event):
        print("\n  --- Step 1: Perception Agent (observing previous round) ---")
        if not dry_run and prev_round_dir.exists():
            from agents.perception_agent import run_perception_agent
            result = run_perception_agent(prev_round_dir, api_key, model, temperature=0.3)
            guard_path = prev_round_dir / "perception_guard.json"
            if guard_path.exists():
                g = json.loads(guard_path.read_text("utf-8"))
                if not g.get("passed", True):
                    print(f"  [guard] perception output flagged: {g}")
            ctx["prev_perception_result"] = result
            print(f"  Perception report saved → {prev_round_dir / 'perception_report.md'}")
        elif dry_run:
            print("  [dry-run] Creating placeholder perception report")
            dummy = f"# Perception Report (Dry Run)\n\nBehavior: dry-run placeholder for round {round_num}."
            (prev_round_dir / "perception_report.md").write_text(dummy, encoding="utf-8")
        coordinator.emit("perception.completed", {"round": round_num - 1, "report_len": len(ctx.get("prev_perception_result", ""))})

    # ─── Step 2: Analyst ────────────────────────────────────────────────
    def _on_perception_completed(event: Event):
        print("\n  --- Step 2: Analyst Agent (ReAct loop) ---")
        from agents.analyst_agent import run_analyst_agent
        if not dry_run:
            proposal = run_analyst_agent(prev_round_dir, round_num, memory_system, api_key, model, temperature=0.4)
            analyst_guard = prev_round_dir / "analyst_guard.json"
            if analyst_guard.exists():
                g = json.loads(analyst_guard.read_text("utf-8"))
                if not g.get("passed", True):
                    print("  [guard] analyst output failed zero-shot guard, retry with lower temperature.")
                    proposal = run_analyst_agent(prev_round_dir, round_num, memory_system, api_key, model, temperature=0.2)
            ctx["proposal"] = proposal
            print(f"  Diagnosis: {proposal.get('diagnosis', 'N/A')[:100]}")
            print(f"  Changes: {proposal.get('changed_count', 0)}")
        else:
            print("  [dry-run] Skipping analyst LLM call")
        coordinator.emit("analyst.completed", {"round": round_num, "changed_count": ctx["proposal"].get("changed_count", 0)})

    # ─── Step 2.5: Constraints + Critic (mid/late only) ─────────────────
    def _on_analyst_completed(event: Event):
        role = ctx["role_stage"]
        if role in ("mid", "late"):
            print("\n  --- Step 2.5: Constraints Agent + Critic Agent ---")
            from agents.constraints_agent import run_constraints_agent
            from agents.critic_agent import run_critic_agent
            proposal = ctx["proposal"]
            if not dry_run:
                constraints_report = run_constraints_agent(prev_round_dir)
                ctx["constraints_report"] = constraints_report
                coordinator.emit("constraints.completed", constraints_report)
                critic_report = run_critic_agent(prev_round_dir, proposal, constraints_report)
                ctx["critic_report"] = critic_report
                coordinator.emit("critic.completed", critic_report)
                if critic_report.get("status") == "needs_revision":
                    proposal.setdefault("risk_mitigation", "")
                    proposal["risk_mitigation"] = (proposal["risk_mitigation"] + " | Critic: "+"; ".join(critic_report.get("critic_flags", []))).strip(" |")
                    print("  [feedback-loop] Re-running Analyst with Critic/Constraints feedback...")
                    (prev_round_dir / "critic_feedback.json").write_text(json.dumps({
                        "critic_report": critic_report, "constraints_report": constraints_report,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    try:
                        revised = run_analyst_agent(prev_round_dir, round_num, memory_system, api_key, model, temperature=0.35)
                        revised.setdefault("risk_mitigation", proposal.get("risk_mitigation", ""))
                        ctx["proposal"] = revised
                    except Exception as e:
                        print(f"  [feedback-loop] Analyst revision FAILED: {e}")
                        print("  [feedback-loop] Keeping original proposal and continuing.")
                        ctx["proposal"] = proposal
                coordinator.emit("generator.ready", {"round": round_num})
            else:
                print("  [dry-run] Skipping constraints/critic LLM calls")
                coordinator.emit("generator.ready", {"round": round_num})
        else:
            # Early round: skip constraints/critic, go straight to generator
            print("  (early round, skipping Constraints+Critic)")
            coordinator.emit("generator.ready", {"round": round_num})

    # ─── Step 3: Generator ──────────────────────────────────────────────
    def _on_generator_ready(event: Event):
        print("\n  --- Step 3: Generator Agent ---")
        from agents.generator_agent import run_generator_agent, validate_generated_code
        proposal = ctx["proposal"]

        import json as _json
        constraints = ""
        if exploration_path and exploration_path.exists():
            try:
                exploration_data = _json.loads(exploration_path.read_text("utf-8"))
                constraints = derive_reward_constraints(exploration_data)
            except Exception:
                pass

        code = None
        if not dry_run:
            gen_result = run_generator_agent(prev_round_dir, proposal, memory_system, api_key, model, temperature=0.3, constraints=constraints)
            code, gen_prompt, gen_responses = gen_result
            (output_dir / "generator_prompt.txt").write_text(gen_prompt, encoding="utf-8")
            for i, r in enumerate(gen_responses):
                suffix = "" if i == 0 else f"_retry{i}"
                (output_dir / f"generator_response{suffix}.md").write_text(r, encoding="utf-8")
            if code is None:
                print("  ERROR: Generator agent failed after retries!")
                print("  [feedback-loop] Re-running Analyst with Generator failure context...")
                from agents.analyst_agent import run_analyst_agent
                validation_issues = []
                if gen_responses:
                    from agents.generator_agent import _extract_reward_code, validate_generated_code, validate_proposal_adherence
                    extracted = _extract_reward_code(gen_responses[-1])
                    if extracted:
                        validation_issues.extend(validate_generated_code(extracted))
                        validation_issues.extend(validate_proposal_adherence(extracted, proposal))
                gen_feedback = {
                    "generator_failed": True,
                    "recent_generator_response": gen_responses[-1] if gen_responses else "",
                    "validation_issues": validation_issues,
                    "proposal": proposal,
                }
                evidence_fingerprint = compute_evidence_fingerprint(validation_issues, proposal)
                gen_feedback["evidence_fingerprint"] = evidence_fingerprint

                feedback_path = prev_round_dir / "generator_feedback.json"
                prev_fingerprint = None
                if feedback_path.exists():
                    try:
                        prev_fingerprint = json.loads(feedback_path.read_text("utf-8")).get("evidence_fingerprint")
                    except Exception:
                        prev_fingerprint = None
                feedback_path.write_text(json.dumps(gen_feedback, ensure_ascii=False, indent=2), encoding="utf-8")

                if not should_rerun_analyst(prev_fingerprint, evidence_fingerprint):
                    print("  [feedback-loop] Skipping analyst re-run: no new evidence since previous generator failure.")
                else:
                    try:
                        proposal = run_analyst_agent(prev_round_dir, round_num, memory_system, api_key, model, temperature=0.35)
                        ctx["proposal"] = proposal
                        gen_result = run_generator_agent(prev_round_dir, proposal, memory_system, api_key, model, temperature=0.25, constraints=constraints)
                        code, gen_prompt, gen_responses = gen_result
                    except Exception as e:
                        print(f"  [feedback-loop] Analyst revision for generator failure FAILED: {e}")
                        code = None
                if code is None:
                    print("  Fallback: copying previous round's reward function.")
                    prev_reward = prev_round_dir / "reward_fn_source.py"
                    if prev_reward.exists():
                        code = prev_reward.read_text("utf-8")
                    else:
                        raise RuntimeError("No reward function available and generation failed.")
        else:
            prev_reward = prev_round_dir / "reward_fn_source.py"
            code = prev_reward.read_text("utf-8") if prev_reward.exists() else "# dry-run"
            print("  [dry-run] Skipping generator LLM call")

        issues = validate_generated_code(code)
        if issues:
            print(f"  WARNING: Code validation issues: {', '.join(issues)}")
        else:
            print(f"  Code validated OK.")

        header = f'"""LLM-generated reward function (round {round_num}).\nSource: {output_dir.name}\n"""\n\nimport math\nimport numpy as np\n\n'
        reward_path = output_dir / "reward_fn_source.py"
        reward_path.write_text(header + code + "\n", encoding="utf-8")
        print(f"  Reward source → {reward_path}")
        ctx["code"] = code
        ctx["reward_path"] = reward_path

        if dry_run or skip_train:
            _write_prompt_efficiency_report(prev_round_dir)
            print("\n  [dry-run/skip-train] Stopping before training.")
            ctx["result"] = {"round": round_num, "proposal": proposal, "code": code, "reward_path": reward_path, "trained": False}
            coordinator.emit("iteration.done", {"round": round_num})
        else:
            coordinator.emit("training.start", {"round": round_num})

    # ─── Step 4: Train ──────────────────────────────────────────────────
    def _on_training_start(event: Event):
        print("\n  --- Step 4: Training ---")
        train_script = Path(__file__).resolve().parent / "train.py"
        train_config = config.get("train", {})
        env_id = config.get("env_id", f"UnknownEnv-round{round_num}")
        training_steps = train_config.get("total_timesteps", config.get("total_timesteps", 2_000_000))
        n_envs = train_config.get("n_envs", config.get("n_envs", 8))

        round_config = {}
        if "ppo" in config:
            round_config["ppo"] = dict(config["ppo"])
        else:
            round_config["ppo"] = {}
        for k in ("evaluation", "checkpoint", "gif_steps", "gif_fps", "gif_max_steps",
                  "seed", "device", "normalize", "max_episode_steps"):
            if k in config:
                round_config[k] = config[k]
        round_config["total_timesteps"] = training_steps
        round_config["n_envs"] = n_envs
        round_config_path = output_dir / "config.yaml"
        with open(round_config_path, "w", encoding="utf-8") as f:
            f.write(_safe_dump_config(round_config))

        cmd = [sys.executable, str(train_script), "--env-dir", str(env_dir),
               "--env-id", config.get("env_id", "UnknownEnv"),
               "--config", str(round_config_path), "--run-dir", str(output_dir),
               "--reward-source", str(ctx["reward_path"])]
        if config.get("max_episode_steps"):
            cmd += ["--max-episode-steps", str(config["max_episode_steps"])]

        print(f"  Running: {' '.join(str(c) for c in cmd)}")
        t0 = perf_counter()
        result = _run_subprocess(cmd)
        elapsed = perf_counter() - t0

        if result.returncode != 0:
            print(f"\n  Training FAILED (exit code {result.returncode})")
            (output_dir / "train_error.log").write_text(result.stderr if result.stderr else "Unknown error (no stderr captured)")
            print("  Attempting self-heal...")
            healed = _self_heal(output_dir, prev_round_dir, api_key, model, config, env_dir)
            if healed:
                print("  Self-heal succeeded. Retrying training...")
                result = _run_subprocess(cmd)
                if result.returncode != 0:
                    (output_dir / "train_error.log").write_text(result.stderr if result.stderr else "Unknown error")
                    ctx["result"] = {"round": round_num, "trained": False, "error": result.returncode}
                    coordinator.emit("iteration.done", {"round": round_num, "trained": False})
                    return
            else:
                ctx["result"] = {"round": round_num, "trained": False, "error": result.returncode}
                coordinator.emit("iteration.done", {"round": round_num, "trained": False})
                return

        print(f"  Training complete ({elapsed/60:.1f} min)")
        ctx["elapsed_minutes"] = round(elapsed / 60, 1)
        coordinator.emit("training.completed", {"round": round_num, "elapsed": elapsed})

    # ─── Step 5: Perception on current round ────────────────────────────
    def _on_training_completed(event: Event):
        print("\n  --- Step 5: Perception Agent (analyzing this round) ---")
        from agents.perception_agent import run_perception_agent
        run_perception_agent(output_dir, api_key, model, temperature=0.3)
        print(f"  Perception report saved → {output_dir / 'perception_report.md'}")
        coordinator.emit("post_perception.completed", {"round": round_num})

    # ─── Step 6: Reflection ─────────────────────────────────────────────
    def _on_post_perception_completed(event: Event):
        print("\n  --- Step 6: Reflection Agent ---")
        from agents.reflection_agent import run_reflection_agent
        run_reflection_agent(output_dir, round_num, memory_system, api_key, model, temperature=0.3)
        print(f"  Reflection saved → {output_dir / 'reflection.md'}")
        print(f"  MEMORY.md updated with lesson from round {round_num}")
        ctx["result"] = {
            "round": round_num,
            "proposal": ctx.get("proposal"),
            "code": ctx.get("code"),
            "reward_path": ctx.get("reward_path"),
            "trained": True,
            "elapsed_minutes": ctx.get("elapsed_minutes", 0),
        }
        coordinator.emit("iteration.done", {"round": round_num, "trained": True})

    # ─── Register all event handlers ────────────────────────────────────
    coordinator.on("iteration.start", _on_iteration_start)
    coordinator.on("perception.completed", _on_perception_completed)
    coordinator.on("analyst.completed", _on_analyst_completed)
    coordinator.on("generator.ready", _on_generator_ready)
    coordinator.on("training.start", _on_training_start)
    coordinator.on("training.completed", _on_training_completed)
    coordinator.on("post_perception.completed", _on_post_perception_completed)

    # ─── Start the event loop ──────────────────────────────────────────
    coordinator.emit("iteration.start", {"round": round_num})

    # Return the result set by the final handler
    return ctx.get("result") or {"round": round_num, "trained": False, "error": "event_loop_did_not_complete"}


def _self_heal(output_dir: Path, prev_round_dir: Path,
                api_key: str, model: str, config: dict,
                env_dir: Path = None) -> bool:
    """Self-heal: fix crashed reward function and re-save.

    Args:
        output_dir: Current round directory.
        prev_round_dir: Previous round directory (for original prompt).
        api_key: LLM API key.
        model: LLM model name.
        config: Experiment config dict.
        env_dir: Environment directory (for signature validation). If None,
                 inferred from config or experiment layout.

    Returns:
        True if heal succeeded, False otherwise.
    """
    from self_heal import build_fix_prompt, validate_signature

    # Determine expected signature from step.py
    expected_signature = "action"  # default fallback
    if env_dir and (env_dir / "step.py").exists():
        step_source = (env_dir / "step.py").read_text("utf-8")
        sig_match = re.search(r'self\.compute_reward\(([^)]+)\)', step_source)
        if sig_match:
            expected_signature = sig_match.group(1).strip()
    elif config.get("env_id"):
        # Try to find env dir from env_id hint
        env_name = re.sub(r"-round\d+$", "", config["env_id"])
        candidate = Path(__file__).resolve().parent.parent / "envs" / env_name / "step.py"
        if candidate.exists():
            sig_match = re.search(r'self\.compute_reward\(([^)]+)\)', candidate.read_text("utf-8"))
            if sig_match:
                expected_signature = sig_match.group(1).strip()

    # Read the failing code
    reward_path = output_dir / "reward_fn_source.py"
    if not reward_path.exists():
        return False

    failing_code = reward_path.read_text("utf-8")

    # Look for error log
    error_log = output_dir / "train_error.log"
    traceback_text = ""
    if error_log.exists():
        traceback_text = error_log.read_text("utf-8").strip()

    if not traceback_text:
        traceback_text = "Training crashed (unknown error)"

    # Build fix prompt (use previous round's prompt as "original")
    prev_prompt = prev_round_dir / "prompt.txt"
    original_prompt = "See reward function source."
    if prev_prompt.exists():
        original_prompt = prev_prompt.read_text("utf-8")

    prompt = build_fix_prompt(original_prompt, failing_code, traceback_text,
                               expected_signature=expected_signature)
    response = call_llm(prompt, api_key, model, temperature=0.4)

    from llm_call import extract_reward_fn
    try:
        fixed_code = extract_reward_fn(response)

        # Validate signature first
        sig_issue = validate_signature(fixed_code, expected_signature)
        if sig_issue:
            print(f"  Self-heal signature mismatch: {sig_issue}")
            return False

        # Validate code structure
        from agents.generator_agent import validate_generated_code
        issues = validate_generated_code(fixed_code)
        if issues:
            print(f"  Self-heal validation issues: {issues}")
            return False

        # Save fixed code
        header = f'"""LLM-generated reward function (self-healed).\nSource: {output_dir.name}\n"""\n\nimport math\nimport numpy as np\n\n'
        reward_path.write_text(header + fixed_code + "\n", encoding="utf-8")
        print(f"  Self-heal: signature={expected_signature}, code validated OK")
        return True
    except Exception as e:
        print(f"  Self-heal extraction failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Reward Function Iteration Pipeline"
    )
    parser.add_argument("--mode", required=True,
                        choices=["round0", "iterate", "continue", "full"],
                        help="round0 = initial generation; iterate = single multi-agent round; continue = auto run remaining rounds; full = round0→train→all rounds")
    parser.add_argument("--experiment-dir", help="Experiment directory (for iterate/continue mode)")
    parser.add_argument("--env-dir", help="Environment directory")
    parser.add_argument("--exploration", help="Exploration JSON path (for round0)")
    parser.add_argument("--config", default=None, help="Experiment configuration YAML")
    parser.add_argument("--round", type=int, default=1, help="Round number to run")
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true", help="Generate prompts only")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = _safe_load_config_text(f.read())
        api_key = cfg.get("llm_api_key", api_key)
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    if args.mode == "round0":
        if not all([args.env_dir, args.exploration]):
            parser.error("--mode round0 requires --env-dir, --exploration, and --config")
        env_dir = Path(args.env_dir).resolve()
        exploration_path = Path(args.exploration).resolve()
        config_path = Path(args.config).resolve() if args.config else None

        # Determine experiment directory
        env_name = env_dir.name
        env_description = _load_env_description(env_name)
        timestamp = datetime.now(BEIJING).strftime("%y%m%d%H%M")
        config_data = _safe_load_config_text(config_path.read_text("utf-8")) if config_path else {}
        total_steps = config_data.get("total_timesteps", "unknown")
        exp_dir = Path(__file__).resolve().parent.parent / "runs" / f"{env_name.lower()}_{timestamp}_{total_steps}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        output_dir = exp_dir / "round0"

        result = run_round0(
            env_dir, exploration_path, config_path,
            output_dir, api_key, args.model, args.temperature, args.dry_run,
            task_description=env_description,
        )

        # Initialize task manifest
        if not args.dry_run:
            memory = MemorySystem(exp_dir)
            step_source = (env_dir / "step.py").read_text("utf-8") if (env_dir / "step.py").exists() else ""
            memory.initialize_task_manifest(step_source=step_source,
                                            env_description=env_description)

        print(f"\nExperiment directory: {exp_dir}")

    elif args.mode == "iterate":
        if not args.experiment_dir:
            parser.error("--mode iterate requires --experiment-dir")
        exp_dir = Path(args.experiment_dir).resolve()
        tee = _setup_logging(exp_dir)

        # Load config first so env_id is available for env dir matching
        config_data = {}
        if args.config:
            config_data = _safe_load_config_text(Path(args.config).read_text("utf-8"))
        else:
            config_data = _load_experiment_config(exp_dir)

        env_dir = Path(args.env_dir).resolve() if args.env_dir else _find_env_dir(exp_dir, config_data.get("env_id"))
        exploration_path = Path(args.exploration).resolve() if args.exploration else _find_exploration(exp_dir)

        # Save experiment-level config if not already present
        exp_config = exp_dir / "config.yaml"
        if not exp_config.exists():
            with open(exp_config, "w", encoding="utf-8") as f:
                f.write(_safe_dump_config(config_data))

        result = run_iteration(
            exp_dir, env_dir, args.round, exploration_path,
            config_data, api_key, args.model, args.temperature,
            args.dry_run, args.skip_train,
        )

    elif args.mode == "continue":
        """Auto-run all remaining rounds in sequence."""
        if not args.experiment_dir:
            parser.error("--mode continue requires --experiment-dir")
        exp_dir = Path(args.experiment_dir).resolve()
        tee = _setup_logging(exp_dir)

        # Load config first so env_id is available for env dir matching
        config_data = {}
        if args.config:
            config_data = _safe_load_config_text(Path(args.config).read_text("utf-8"))
        else:
            config_data = _load_experiment_config(exp_dir)

        env_dir = Path(args.env_dir).resolve() if args.env_dir else _find_env_dir(exp_dir, config_data.get("env_id"))
        exploration_path = Path(args.exploration).resolve() if args.exploration else _find_exploration(exp_dir)

        # Save experiment-level config if not already present
        exp_config = exp_dir / "config.yaml"
        if not exp_config.exists():
            with open(exp_config, "w", encoding="utf-8") as f:
                f.write(_safe_dump_config(config_data))

        total_rounds = config_data.get("rounds", 5)
        memory = MemorySystem(exp_dir)
        existing_rounds = memory.get_available_rounds()
        start_round = max(existing_rounds) + 1 if existing_rounds else 1

        # Patch missing reflections in existing rounds before continuing
        for r in existing_rounds:
            if r == 0:
                continue  # round0 never has reflection
            _patch_missing_reflection(
                exp_dir / f"round{r}", r, memory,
                api_key, args.model, args.temperature,
            )

        for r in range(start_round, total_rounds + 1):
            print(f"\n{'#'*60}")
            print(f"  Auto-continue: Round {r}")
            print(f"{'#'*60}")
            result = run_iteration(
                exp_dir, env_dir, r, exploration_path,
                config_data, api_key, args.model, args.temperature,
                dry_run=False, skip_train=False,
            )
            if not result.get("trained", False):
                print(f"  Round {r} failed to train. Stopping.")
                break

    elif args.mode == "full":
        if not all([args.env_dir, args.exploration, args.config]):
            parser.error("--mode full requires --env-dir, --exploration, and --config")

        env_dir = Path(args.env_dir).resolve()
        exploration_path = Path(args.exploration).resolve()
        config_path = Path(args.config).resolve()
        config_data = _safe_load_config_text(config_path.read_text("utf-8"))
        env_name = env_dir.name
        env_description = _load_env_description(env_name)
        total_rounds = config_data.get("rounds", 5)

        # Create experiment directory
        timestamp = datetime.now(BEIJING).strftime("%y%m%d%H%M")
        total_steps = config_data.get("total_timesteps", "unknown")
        exp_dir = Path(__file__).resolve().parent.parent / "runs" / f"{env_name.lower()}_{timestamp}_{total_steps}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        tee = _setup_logging(exp_dir)

        print(f"\n{'='*60}")
        print(f"  Full Auto Mode: {env_name}")
        print(f"  Experiment: {exp_dir}")
        print(f"  Rounds: 0..{total_rounds}")
        print(f"{'='*60}\n")

        if args.dry_run:
            result = run_round0(env_dir, exploration_path, config_path, exp_dir / "round0",
                                api_key, args.model, args.temperature, dry_run=True,
                                task_description=env_description)
            print(f"  [dry-run] Experiment directory: {exp_dir}")
            return

        # ── Phase 1: Round 0 — generate initial reward ──
        print(">>> Phase 1/4: Generating initial reward function")
        round0_result = run_round0(env_dir, exploration_path, config_path, exp_dir / "round0",
                                   api_key, args.model, args.temperature,
                                   task_description=env_description)

        # Save experiment-level config (without API key)
        public_config = {k: v for k, v in config_data.items() if k != "llm_api_key"}
        (exp_dir / "config.yaml").write_text(_safe_dump_config(public_config), encoding="utf-8")

        # ── Phase 2: Train round 0 ──
        print("\n>>> Phase 2/4: Training Round 0")
        train_script = Path(__file__).resolve().parent / "train.py"
        round0_dir = exp_dir / "round0"

        # Full config (without API key) goes into round0 for train.py
        round0_config_path = round0_dir / "config.yaml"
        (round0_config_path).write_text(_safe_dump_config(public_config), encoding="utf-8")

        cmd = [
            sys.executable, str(train_script),
            "--env-dir", str(env_dir),
            "--env-id", f"{env_name}-round0",
            "--config", str(round0_config_path),
            "--run-dir", str(round0_dir),
            "--reward-source", str(round0_dir / "reward_fn_source.py"),
        ]
        if config_data.get("max_episode_steps"):
            cmd += ["--max-episode-steps", str(config_data["max_episode_steps"])]

        print(f"  Running: {' '.join(str(c) for c in cmd)}")
        t0 = perf_counter()
        train_r0 = _run_subprocess(cmd)
        t_elapsed = perf_counter() - t0

        if train_r0.returncode != 0:
            print(f"  Round 0 training FAILED after {t_elapsed/60:.1f} min. Aborting.")
            sys.exit(1)
        print(f"  Round 0 training complete ({t_elapsed/60:.1f} min)")

        # ── Phase 3: Initialize memory & task manifest ──
        print("\n>>> Phase 3/4: Initializing memory system")
        memory = MemorySystem(exp_dir)
        step_source = (env_dir / "step.py").read_text("utf-8") if (env_dir / "step.py").exists() else ""
        memory.initialize_task_manifest(step_source=step_source,
                                        env_description=env_description)
        print(f"  Task manifest → {memory.task_manifest_path}")

        # ── Phase 4: Iterate rounds 1..N ──
        print(f"\n>>> Phase 4/4: Running rounds 1..{total_rounds}")
        config_data["env_id"] = f"{env_name}-round0"

        for r in range(1, total_rounds + 1):
            print(f"\n{'#'*60}")
            print(f"  Iteration Round {r}")
            print(f"{'#'*60}")
            iter_result = run_iteration(
                exp_dir, env_dir, r, exploration_path,
                config_data, api_key, args.model, args.temperature,
                dry_run=False, skip_train=False,
            )
            if not iter_result.get("trained", False):
                print(f"  Round {r} FAILED. Stopping.")
                (exp_dir / "STATUS").write_text(f"FAILED at round {r}\n")
                break

        (exp_dir / "STATUS").write_text(f"COMPLETED ({total_rounds} rounds)\n")
        print(f"\n{'='*60}")
        print(f"  Experiment complete!")
        print(f"  Directory: {exp_dir}")
        print(f"{'='*60}")


def _find_env_dir(exp_dir: Path, env_id_hint: str = None) -> Path:
    """Find the env directory matching the experiment context.

    Args:
        exp_dir: Experiment directory (name encodes env name).
        env_id_hint: Optional env_id (e.g. 'BipedalWalker-v3-round0') to match.

    Returns:
        Path to the environment directory.

    Raises:
        FileNotFoundError: No matching env directory found.
    """
    envs_dir = exp_dir.parent.parent / "envs"
    if not envs_dir.exists():
        raise FileNotFoundError(f"envs directory not found: {envs_dir}")

    # Method 1: Match by env_id_hint (most reliable)
    if env_id_hint:
        import re
        env_name = re.sub(r"-round\d+$", "", env_id_hint)
        for d in envs_dir.iterdir():
            if d.is_dir() and (d / "step.py").exists() and d.name == env_name:
                return d
        # Case-insensitive fallback
        for d in envs_dir.iterdir():
            if d.is_dir() and (d / "step.py").exists() and d.name.lower() == env_name.lower():
                return d

    # Method 2: Parse env name from experiment directory name (format: envname_timestamp_timesteps)
    exp_name = exp_dir.name
    env_name = exp_name.rsplit("_", 2)[0]
    for d in envs_dir.iterdir():
        if d.is_dir() and (d / "step.py").exists() and d.name.lower() == env_name.lower():
            return d

    # Method 3: Fallback — first env dir with step.py (original behavior)
    for d in sorted(envs_dir.iterdir()):
        if d.is_dir() and (d / "step.py").exists():
            print(f"  WARNING: Could not match env by name, using first found: {d.name}")
            return d

    raise FileNotFoundError(
        "Cannot find env directory from experiment context. "
        "Use --env-dir to specify it explicitly."
    )


def _find_exploration(exp_dir: Path) -> Path:
    """Try to find exploration JSON."""
    explores_dir = exp_dir.parent.parent / "explorations"
    if explores_dir.exists():
        for f in explores_dir.glob("*.json"):
            return f
    return None


def _load_experiment_config(exp_dir: Path) -> dict:
    """Load config from existing experiment files."""
    round0_dir = exp_dir / "round0"
    if round0_dir.exists():
        config_path = round0_dir / "config.yaml"
        if config_path.exists():
            return _safe_load_config_text(config_path.read_text("utf-8"))
    return {
        "total_timesteps": 2_000_000,
        "n_envs": 8,
        "rounds": 5,
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    }


if __name__ == "__main__":
    main()
