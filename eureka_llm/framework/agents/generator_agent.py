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
from prompt_compaction import load_prompt_policy, summarize_structured_lines, write_compaction_stats
from prompt_harness import build_contract_block


def build_generator_prompt(run_dir: Path, proposal: dict,
                            memory_system,
                            constraints: str = "") -> tuple[str, dict]:
    """Build prompt for the generator agent."""
    template_path = Path(__file__).resolve().parent.parent.parent / "templates" / "generator_prompt.txt"
    template = template_path.read_text("utf-8") if template_path.exists() else _fallback_generator_prompt()

    # Load current reward function
    current_reward = ""
    reward_path = run_dir / "reward_fn_source.py"
    if reward_path.exists():
        current_reward = reward_path.read_text("utf-8")

    # Load prompt compaction policy
    policy = load_prompt_policy(run_dir.parent if run_dir.name.startswith("round") else run_dir, "generator")
    stats = {}

    # Load task manifest
    task_manifest = memory_system.get_task_manifest()
    if task_manifest:
        task_manifest, stats["task_manifest"] = summarize_structured_lines(
            task_manifest, policy.get("max_lines_markdown", 80), ("task", "termination", "observation")
        )

    # Load perception report
    perception = ""
    perception_path = run_dir / "perception_report.md"
    if perception_path.exists():
        perception, stats["perception_report"] = summarize_structured_lines(
            perception_path.read_text("utf-8"), policy.get("max_lines_markdown", 80), ("diagnosis", "principle", "mean", "std")
        )

    # Build the sections
    contract = build_contract_block(
        agent="Generator",
        objective="Translate analyst proposal into runnable reward code with no drift.",
        required_outputs=[
            "Python code with compute_reward and metrics_fn",
            "All analyst new_code edits applied",
            "No simulator object persistence anti-patterns",
        ],
        hard_constraints=[
            "Do not introduce unrelated reward terms",
            "Keep signatures and return contracts valid",
            "If proposal conflicts exist, choose explicit proposal edits",
        ],
    )

    sections = [
        contract,
        "",
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

    if perception:
        sections.append("")
        sections.append("## Perception Report (for context)")
        sections.append(perception)
    if constraints:
        sections.append("")
        sections.append("## Environment Constraints")
        sections.append(constraints)

    return "\n".join(sections), stats


def validate_proposal_adherence(code: str, proposal: dict) -> list[str]:
    """Lightweight check that generated code contains analyst-requested code edits."""
    issues = []
    for change in proposal.get("proposed_changes", []) or []:
        new_code = (change.get("new_code") or "").strip()
        if not new_code:
            continue
        target = new_code.split("#", 1)[0].strip()
        if not target:
            continue
        if target not in code:
            component = change.get("component", "unknown_component")
            issues.append(f"Proposal change not applied: {component} -> `{target}`")
            continue

        # Semantic-ish assignment check: if new_code has `lhs = rhs`, ensure rhs survived
        m_new = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*([^#\n]+)", target)
        if m_new:
            lhs, rhs = m_new.group(1).strip(), m_new.group(2).strip()
            assign_pattern = rf"{re.escape(lhs)}\s*=\s*{re.escape(rhs)}(?:\s|$)"
            if not re.search(assign_pattern, code):
                issues.append(f"Assignment mismatch for `{lhs}`: expected `{rhs}`")

        # If current_code is explicit, ensure obvious anti-pattern is removed
        old_code = (change.get("current_code") or "").split("#", 1)[0].strip()
        if old_code and old_code != target and old_code in code:
            issues.append(f"Old pattern still present for `{change.get('component', 'unknown_component')}`: `{old_code}`")
    return issues


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
                         max_retries: int = 2,
                         constraints: str = "") -> tuple[Optional[str], str, list]:
    """Run the generator agent to produce validated reward function code.

    Args:
        run_dir: Base run directory (for loading current reward function)
        proposal: Structured proposal from analyst agent
        memory_system: Memory system for task manifest
        api_key: LLM API key
        model: Model name
        temperature: Lower temperature for code generation
        max_retries: How many times to retry if validation fails
        constraints: Environment-specific reward constraints (from exploration)

    Returns:
        Validated code string, or None if all retries failed
    """
    prompt, compaction_stats = build_generator_prompt(run_dir, proposal, memory_system, constraints)
    write_compaction_stats(run_dir / "generator_prompt_compaction.json", compaction_stats)
    print(f"  Generator prompt: {len(prompt)} chars")
    all_responses = []

    for attempt in range(max_retries + 1):
        print(f"  LLM call attempt {attempt + 1}/{max_retries + 1} ...")
        response = call_llm(prompt, api_key, model, temperature, timeout=600.0)
        all_responses.append(response)

        # Extract code
        code = _extract_reward_code(response)
        if not code:
            if attempt < max_retries:
                prompt += "\n\n[RETRY: Previous response had no valid code block. Output only ```python block.]"
                continue
            return None, prompt, all_responses

        issues = validate_generated_code(code)
        issues.extend(validate_proposal_adherence(code, proposal))
        if not issues:
            try:
                memory_system.update_belief("generator", {
                    "round": run_dir.name,
                    "status": "success",
                    "attempt": attempt + 1,
                    "changed_count": proposal.get("changed_count", 0),
                })
            except Exception:
                pass
            return code, prompt, all_responses

        # If validation failed and we have retries left, add issues to prompt
        if attempt < max_retries:
            issue_str = "; ".join(issues)
            prompt += f"\n\n[RETRY: Previous code had issues: {issue_str}. Fix ALL issues.]"

    try:
        memory_system.update_belief("generator", {
            "round": run_dir.name,
            "status": "failed",
            "attempts": max_retries + 1,
            "changed_count": proposal.get("changed_count", 0),
        })
    except Exception:
        pass
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
