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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter

import yaml

# Ensure framework directory is on path for imports
_framework_dir = Path(__file__).resolve().parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from llm_call import call_llm, extract_reward_fn, save_artifacts
from template_engine import (
    build_round0_prompt,
    load_training_data,
)
from memory.memory_system import MemorySystem


BEIJING = timezone(timedelta(hours=8))


class _Tee:
    """Tee stdout to both console and a log file."""
    def __init__(self, log_path: Path):
        self.log = log_path.open("w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, data: str):
        self.stdout.write(data)
        self.log.write(data)

    def flush(self):
        self.stdout.flush()
        self.log.flush()


def _setup_logging(exp_dir: Path) -> _Tee:
    """Redirect stdout to tee into experiment.log."""
    tee = _Tee(exp_dir / "experiment.log")
    sys.stdout = tee
    return tee


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


# ─────────────────────────────────────────────────────────────────────────────
# Round 0
# ─────────────────────────────────────────────────────────────────────────────

def run_round0(env_dir: Path, exploration_path: Path, config_path: Path,
               output_dir: Path, api_key: str, model: str = "deepseek-reasoner",
               temperature: float = 0.6, dry_run: bool = False) -> dict:
    """Round 0: generate initial reward function from environment exploration.

    This is the same as the original round0 flow, but additionally creates
    the task manifest for future rounds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build round0 prompt (same as original)
    template_path = Path(__file__).resolve().parent.parent / "templates" / "round0_prompt.txt"
    prompt = build_round0_prompt(env_dir, template_path, exploration_path)

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
    """Run one multi-agent iteration round.

    Full agent workflow:
        1. Perception Agent → perception_report.md (analyze previous training)
        2. Analyst Agent (ReAct) → analyst_proposal.json (structured proposal)
        3. Generator Agent → reward_fn_source.py (validated code)
        4. Self-Heal Check: if train crashes, auto-fix and retry
        5. Train → new policy
        6. Perception Agent → perception_report.md (analyze this round)
        7. Reflection Agent → reflection.md + MEMORY.md update

    Args:
        run_dir: Base experiment directory (contains round0/, round1/, etc.)
        env_dir: Environment directory (env.py, step.py)
        round_num: Current round number (1, 2, 3, ...)
        exploration_path: Path to exploration JSON for action/obs summaries
        config: Full experiment config dict
        api_key: LLM API key
        model: Model name for LLM calls
        temperature: Temperature for LLM calls
        dry_run: If True, only generate prompts without calling LLMs
        skip_train: If True, skip training (for debugging agent workflow)

    Returns:
        Dict with iteration results
    """
    prev_round_dir = run_dir / f"round{round_num - 1}"
    output_dir = run_dir / f"round{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_system = MemorySystem(run_dir)

    # Load previous round's perception report for analyst prompt
    prev_perception_path = prev_round_dir / "perception_report.md"
    prev_perception = ""
    if prev_perception_path.exists():
        prev_perception = prev_perception_path.read_text("utf-8")

    # Load previous round's training data for context
    prev_data = load_training_data(prev_round_dir)

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Iteration — Round {round_num}")
    print(f"  Run dir: {run_dir.name}")
    print(f"  Previous: round{round_num - 1}")
    print(f"{'='*60}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Perception Agent — analyze previous round's training
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- Step 1: Perception Agent (observing previous round) ---")
    from agents.perception_agent import run_perception_agent

    prev_perception_result = ""
    if not dry_run and prev_round_dir.exists():
        prev_perception_result = run_perception_agent(
            prev_round_dir, api_key, model, temperature=0.3,
        )
        print(f"  Perception report saved → {prev_round_dir / 'perception_report.md'}")
    elif dry_run:
        print("  [dry-run] Creating placeholder perception report")
        dummy_report = f"# Perception Report (Dry Run)\n\nBehavior: dry-run placeholder for round {round_num}."
        (prev_round_dir / "perception_report.md").write_text(dummy_report, encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Analyst Agent (ReAct loop) — diagnose and propose
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- Step 2: Analyst Agent (ReAct loop) ---")
    from agents.analyst_agent import run_analyst_agent

    if not dry_run:
        proposal = run_analyst_agent(
            prev_round_dir, round_num, memory_system,
            api_key, model, temperature=0.4,
        )
        print(f"  Proposal saved → {prev_round_dir / 'analyst_proposal.json'}")
        print(f"  Diagnosis: {proposal.get('diagnosis', 'N/A')[:100]}")
        print(f"  Changes: {proposal.get('changed_count', 0)}")
    else:
        proposal = {"diagnosis": "dry-run", "changed_count": 0, "proposed_changes": []}
        print("  [dry-run] Skipping analyst LLM call")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Generator Agent — generate validated reward code
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- Step 3: Generator Agent ---")
    from agents.generator_agent import run_generator_agent, validate_generated_code

    if not dry_run:
        gen_result = run_generator_agent(
            prev_round_dir, proposal, memory_system,
            api_key, model, temperature=0.3,
        )
        code, gen_prompt, gen_responses = gen_result
        # Save generator artifacts in output dir
        (output_dir / "generator_prompt.txt").write_text(gen_prompt, encoding="utf-8")
        for i, r in enumerate(gen_responses):
            suffix = "" if i == 0 else f"_retry{i}"
            (output_dir / f"generator_response{suffix}.md").write_text(r, encoding="utf-8")
        if code is None:
            print("  ERROR: Generator agent failed to produce valid code after retries!")
            print("  Falling back to copying previous round's reward function.")
            prev_reward = prev_round_dir / "reward_fn_source.py"
            if prev_reward.exists():
                code = prev_reward.read_text("utf-8")
            else:
                raise RuntimeError("No reward function available and generation failed.")
    else:
        # Dry run: copy previous reward
        prev_reward = prev_round_dir / "reward_fn_source.py"
        code = prev_reward.read_text("utf-8") if prev_reward.exists() else "# dry-run"
        print("  [dry-run] Skipping generator LLM call")

    # Validate one more time and save
    issues = validate_generated_code(code)
    if issues:
        print(f"  WARNING: Code validation issues: {', '.join(issues)}")
    else:
        print(f"  Code validated OK.")

    header = f'"""LLM-generated reward function (round {round_num}).\nSource: {output_dir.name}\n"""\n\nimport math\nimport numpy as np\n\n'
    reward_path = output_dir / "reward_fn_source.py"
    reward_path.write_text(header + code + "\n", encoding="utf-8")
    print(f"  Reward source → {reward_path}")

    if dry_run or skip_train:
        print("\n  [dry-run/skip-train] Stopping before training.")
        return {
            "round": round_num,
            "proposal": proposal,
            "code": code,
            "reward_path": reward_path,
            "trained": False,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Train
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- Step 4: Training ---")
    train_config = config.get("train", {})
    train_script = Path(__file__).resolve().parent / "train.py"

    env_id = config.get("env_id", f"UnknownEnv-round{round_num}")
    training_steps = train_config.get("total_timesteps", config.get("total_timesteps", 2_000_000))
    n_envs = train_config.get("n_envs", config.get("n_envs", 8))

    # Prepare config for this round (preserve ppo nested + eval/checkpoint/gif settings)
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
        yaml.safe_dump(round_config, f, sort_keys=False)

    # Build CLI args for train.py
    cmd = [
        sys.executable, str(train_script),
        "--env-dir", str(env_dir),
        "--env-id", config.get("env_id", "UnknownEnv"),
        "--config", str(round_config_path),
        "--run-dir", str(output_dir),
        "--reward-source", str(reward_path),
    ]
    if config.get("max_episode_steps"):
        cmd += ["--max-episode-steps", str(config["max_episode_steps"])]

    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    t0 = perf_counter()

    result = _run_subprocess(cmd)
    elapsed = perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  Training FAILED (exit code {result.returncode})")
        # Save error log for self-heal analysis
        (output_dir / "train_error.log").write_text(
            result.stderr if result.stderr else "Unknown error (no stderr captured)"
        )
        # ── Self-Heal ──
        print("  Attempting self-heal...")
        healed = _self_heal(output_dir, prev_round_dir, api_key, model, config, env_dir)
        if healed:
            print("  Self-heal succeeded. Retrying training...")
            result = _run_subprocess(cmd)
            if result.returncode != 0:
                print(f"  Retry FAILED. Error log saved.")
                (output_dir / "train_error.log").write_text(result.stderr if result.stderr else "Unknown error")
                return {"round": round_num, "trained": False, "error": result.returncode}
        else:
            print("  Self-heal failed. Check the error log.")
            return {"round": round_num, "trained": False, "error": result.returncode}

    print(f"  Training complete ({elapsed/60:.1f} min)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Perception Agent — analyze THIS round's results
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- Step 5: Perception Agent (analyzing this round) ---")
    perception_result = run_perception_agent(
        output_dir, api_key, model, temperature=0.3,
    )
    print(f"  Perception report saved → {output_dir / 'perception_report.md'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Reflection Agent — generate causal lesson
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- Step 6: Reflection Agent ---")
    from agents.reflection_agent import run_reflection_agent
    reflection = run_reflection_agent(
        output_dir, round_num, memory_system,
        api_key, model, temperature=0.3,
    )
    print(f"  Reflection saved → {output_dir / 'reflection.md'}")
    print(f"  MEMORY.md updated with lesson from round {round_num}")

    return {
        "round": round_num,
        "proposal": proposal,
        "code": code,
        "reward_path": reward_path,
        "trained": True,
        "elapsed_minutes": round(elapsed / 60, 1),
    }


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
            cfg = yaml.safe_load(f)
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
        timestamp = datetime.now(BEIJING).strftime("%y%m%d%H%M")
        config_data = yaml.safe_load(config_path.read_text("utf-8")) if config_path else {}
        total_steps = config_data.get("total_timesteps", "unknown")
        exp_dir = Path(__file__).resolve().parent.parent / "runs" / f"{env_name.lower()}_{timestamp}_{total_steps}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        output_dir = exp_dir / "round0"

        result = run_round0(
            env_dir, exploration_path, config_path,
            output_dir, api_key, args.model, args.temperature, args.dry_run,
        )

        # Initialize task manifest
        if not args.dry_run:
            memory = MemorySystem(exp_dir)
            step_source = (env_dir / "step.py").read_text("utf-8") if (env_dir / "step.py").exists() else ""
            memory.initialize_task_manifest(step_source=step_source)

        print(f"\nExperiment directory: {exp_dir}")

    elif args.mode == "iterate":
        if not args.experiment_dir:
            parser.error("--mode iterate requires --experiment-dir")
        exp_dir = Path(args.experiment_dir).resolve()
        tee = _setup_logging(exp_dir)

        # Load config first so env_id is available for env dir matching
        config_data = {}
        if args.config:
            config_data = yaml.safe_load(Path(args.config).read_text("utf-8"))
        else:
            config_data = _load_experiment_config(exp_dir)

        env_dir = Path(args.env_dir).resolve() if args.env_dir else _find_env_dir(exp_dir, config_data.get("env_id"))
        exploration_path = Path(args.exploration).resolve() if args.exploration else _find_exploration(exp_dir)

        # Save experiment-level config if not already present
        exp_config = exp_dir / "config.yaml"
        if not exp_config.exists():
            with open(exp_config, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, sort_keys=False)

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
            config_data = yaml.safe_load(Path(args.config).read_text("utf-8"))
        else:
            config_data = _load_experiment_config(exp_dir)

        env_dir = Path(args.env_dir).resolve() if args.env_dir else _find_env_dir(exp_dir, config_data.get("env_id"))
        exploration_path = Path(args.exploration).resolve() if args.exploration else _find_exploration(exp_dir)

        # Save experiment-level config if not already present
        exp_config = exp_dir / "config.yaml"
        if not exp_config.exists():
            with open(exp_config, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, sort_keys=False)

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
        config_data = yaml.safe_load(config_path.read_text("utf-8"))
        env_name = env_dir.name
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
                                api_key, args.model, args.temperature, dry_run=True)
            print(f"  [dry-run] Experiment directory: {exp_dir}")
            return

        # ── Phase 1: Round 0 — generate initial reward ──
        print(">>> Phase 1/4: Generating initial reward function")
        round0_result = run_round0(env_dir, exploration_path, config_path, exp_dir / "round0",
                                   api_key, args.model, args.temperature)

        # Save experiment-level config (without API key)
        public_config = {k: v for k, v in config_data.items() if k != "llm_api_key"}
        (exp_dir / "config.yaml").write_text(yaml.safe_dump(public_config, sort_keys=False), encoding="utf-8")

        # ── Phase 2: Train round 0 ──
        print("\n>>> Phase 2/4: Training Round 0")
        train_script = Path(__file__).resolve().parent / "train.py"
        round0_dir = exp_dir / "round0"

        # Full config (without API key) goes into round0 for train.py
        round0_config_path = round0_dir / "config.yaml"
        (round0_config_path).write_text(yaml.safe_dump(public_config, sort_keys=False), encoding="utf-8")

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
        memory.initialize_task_manifest(step_source=step_source)
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
            return yaml.safe_load(config_path.read_text("utf-8"))
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
