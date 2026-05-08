"""
generator_agent.py — Translates analyst proposals into valid Python reward code.

Role in multi-agent system:
    Analyst Proposal → Generator Agent → reward_fn_source.py → Train

Validates:
    1. Has compute_reward function
    2. Has metrics_fn function
    3. No Box2D object storage
    4. Returns (float, dict)
    5. Syntax valid
"""

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Optional
# tuple is a built-in type in Python 3.9+ — no import needed

# Ensure framework directory is on path for imports
_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


GENERATOR_SYSTEM_PROMPT = """You are the Generator Agent — an AI that translates structured proposals
into correct, runnable Python code. You apply precise, targeted changes."""


def build_generator_prompt(run_dir: Path, proposal: dict,
                            memory_system) -> str:
    """Build prompt for the generator agent."""
    template_path = Path(__file__).resolve().parent.parent.parent / "templates" / "generator_prompt.txt"
    template = template_path.read_text("utf-8") if template_path.exists() else _fallback_generator_prompt()

    # Load current reward function
    current_reward = ""
    reward_path = run_dir / "reward_fn_source.py"
    if reward_path.exists():
        current_reward = reward_path.read_text("utf-8")

    # Load task manifest
    task_manifest = memory_system.get_task_manifest()
    if task_manifest:
        # Keep only the first 1000 chars for context
        task_manifest = task_manifest[:2000]

    # Load perception report
    perception = ""
    perception_path = run_dir / "perception_report.md"
    if perception_path.exists():
        perception = perception_path.read_text("utf-8")

    # Build the sections
    sections = [
        template,
        "",
        "## Current Reward Function",
        "```python",
        current_reward,
        "```",
        "",
        "## Analyst Proposal",
        "```json",
        json.dumps(proposal, indent=2, ensure_ascii=False),
        "```",
    ]

    if task_manifest:
        sections.append("")
        sections.append("## Task Manifest (excerpt)")
        sections.append(task_manifest)

    return "\n".join(sections)


def validate_generated_code(code: str) -> list[str]:
    """Validate generated code. Returns list of issues (empty = valid)."""
    issues = []

    # Must have compute_reward
    if "def compute_reward" not in code:
        issues.append("Missing 'def compute_reward'")

    # Must have metrics_fn
    if "def metrics_fn" not in code:
        issues.append("Missing 'def metrics_fn' — required for evaluation")

    # Must have return
    if "return" not in code:
        issues.append("No return statement found")

    # Must have components dict
    if "components" not in code:
        issues.append("No 'components' dict — CARD tracking will be empty")

    # Check for simulator object storage (common crash cause with SubprocVecEnv)
    sim_patterns = [
        r"self\.\w+\s*=\s*self\.\w+\.(position|linearVelocity|angle|angularVelocity)",
        r"self\.\w+\s*=\s*self\.\w+\[\d+\]",
    ]
    for pattern in sim_patterns:
        if re.search(pattern, code):
            issues.append(f"Potential simulator object storage: {pattern}")
            break

    # metrics_fn contract checks
    if "def metrics_fn(env, action)" not in code:
        issues.append("metrics_fn signature must be exactly: def metrics_fn(env, action)")
    if "env.unwrapped" in code:
        issues.append("Do not use env.unwrapped inside metrics_fn; env is already unwrapped")

    # Common anti-pattern: using physics awake/sleep as task metric or success logic
    if re.search(r"\.awake\b|sleep", code):
        issues.append("Avoid awake/sleep engine state in reward/metrics logic")

    # Check syntax
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")

    return issues


def run_generator_agent(run_dir: Path, proposal: dict,
                         memory_system, api_key: str,
                         model: str = "deepseek-reasoner",
                         temperature: float = 0.3,
                         max_retries: int = 2) -> tuple[Optional[str], str, list]:
    """Run the generator agent to produce validated reward function code.

    Args:
        run_dir: Base run directory (for loading current reward function)
        proposal: Structured proposal from analyst agent
        memory_system: Memory system for task manifest
        api_key: LLM API key
        model: Model name
        temperature: Lower temperature for code generation
        max_retries: How many times to retry if validation fails

    Returns:
        Validated code string, or None if all retries failed
    """
    prompt = build_generator_prompt(run_dir, proposal, memory_system)
    print(f"  Generator prompt: {len(prompt)} chars")
    all_responses = []

    for attempt in range(max_retries + 1):
        print(f"  LLM call attempt {attempt + 1}/{max_retries + 1} ...")
        response = call_llm(prompt, api_key, model, temperature, timeout=300.0)
        all_responses.append(response)

        # Extract code
        code = _extract_reward_code(response)
        if not code:
            if attempt < max_retries:
                prompt += "\n\n[RETRY: Previous response had no valid code block. Output only ```python block.]"
                continue
            return None, prompt, all_responses

        issues = validate_generated_code(code)
        if not issues:
            return code, prompt, all_responses

        # If validation failed and we have retries left, add issues to prompt
        if attempt < max_retries:
            issue_str = "; ".join(issues)
            prompt += f"\n\n[RETRY: Previous code had issues: {issue_str}. Fix ALL issues.]"

    return None, prompt, all_responses


def _extract_reward_code(response_text: str) -> Optional[str]:
    """Extract Python code blocks from LLM response."""
    blocks = re.findall(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if blocks:
        # Combine all blocks that contain compute_reward or metrics_fn
        relevant = [b for b in blocks if "def compute_reward" in b or "def metrics_fn" in b]
        if relevant:
            return "\n\n".join(relevant) + "\n"
        return blocks[0] + "\n"
    return None


def _fallback_generator_prompt() -> str:
    """Fallback generator prompt if template file is missing."""
    return """Generate a complete Python module with compute_reward(self, action) and metrics_fn(env, action).

The reward function must return (float, dict).
The metrics_fn must return a dict of task-level metrics.

CRITICAL:
- metrics_fn is MANDATORY
- No Box2D object storage
- All imports inside functions
"""
