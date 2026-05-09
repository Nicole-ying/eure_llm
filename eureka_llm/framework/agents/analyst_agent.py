"""
analyst_agent.py — ReAct-loop agent that diagnoses training problems and proposes
structured reward function changes.

Architecture:
    ReAct loop: Thought → Action (tool call) → Observation → Thought → Final Proposal

Tools available:
    - query_memory(keyword): Search cross-round lessons
    - calculate_reward_budget(changes_summary): Predict net reward effect
    - compare_rounds(n, m): Contrast two rounds

Output:
    Structured JSON proposal with changed_count ≤ 3
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
from memory.memory_system import MemorySystem

ANALYST_SYSTEM_PROMPT = """You are the Analyst Agent — the core reasoning engine of a multi-agent
reward function design system. You use a Thought → Action → Observation loop
to diagnose training problems and produce structured proposals."""


def build_analyst_prompt(run_dir: Path, round_num: int,
                          memory_system: MemorySystem) -> str:
    """Build the full analyst prompt with context from multiple sources."""
    sys_prompt_path = Path(__file__).resolve().parent.parent.parent / "templates" / "analyst_system_prompt.txt"
    sys_prompt = sys_prompt_path.read_text("utf-8") if sys_prompt_path.exists() else ANALYST_SYSTEM_PROMPT

    # Load perception report
    perception_report = ""
    perception_path = run_dir / "perception_report.md"
    if perception_path.exists():
        perception_report = perception_path.read_text("utf-8")

    # Load current reward function
    current_reward = ""
    reward_path = run_dir / "reward_fn_source.py"
    if reward_path.exists():
        src = reward_path.read_text("utf-8")
        # Truncate to keep prompt manageable (show first 150 lines)
        lines = src.splitlines()
        current_reward = "\n".join(lines[:150])
        if len(lines) > 150:
            current_reward += f"\n# ... ({len(lines) - 150} more lines)"

    # Load task manifest
    task_manifest = memory_system.get_task_manifest()
    # Truncate step.py to avoid prompt overflow
    step_match = re.search(r"```python\n(.*?)```", task_manifest, re.DOTALL)
    if step_match:
        step_lines = step_match.group(1).splitlines()
        task_manifest_short = task_manifest[:1000]  # Keep first 1000 chars
    else:
        task_manifest_short = task_manifest[:1000]

    # Load cross-round memory
    lessons = memory_system.get_lessons(max_lines=200)
    recent_history = memory_system.get_recent_lessons(n=3)

    # Load last round's analyst proposal (if any)
    prev_proposal = ""
    prev_round_dir = run_dir.parent / f"round{round_num - 1}" if round_num > 0 else None
    if prev_round_dir and prev_round_dir.exists():
        proposal_path = prev_round_dir / "analyst_proposal.json"
        if proposal_path.exists():
            prev_proposal = proposal_path.read_text("utf-8")

    # Build the prompt
    sections = [
        sys_prompt,
        "",
        "---",
        "## Current Context",
        "",
        f"### Current Round: {round_num}",
        f"### Previous Round Results: run_dir = {run_dir.name}",
        "",
    ]

    if task_manifest_short:
        sections.append("### Task Manifest (excerpt)")
        sections.append(task_manifest_short)
        sections.append("")

    if perception_report:
        sections.append("### Perception Report")
        sections.append(perception_report)
        sections.append("")

    if current_reward:
        sections.append("### Current Reward Function")
        sections.append("```python")
        sections.append(current_reward)
        sections.append("```")
        sections.append("")

    if prev_proposal:
        sections.append("### Previous Round's Proposal (what we predicted)")
        sections.append("```json")
        sections.append(prev_proposal)
        sections.append("```")
        sections.append("")

    if recent_history:
        sections.append("### Recent Round History")
        sections.append(recent_history)
        sections.append("")

    if lessons:
        sections.append("### Cross-Round Lessons (MEMORY.md)")
        sections.append(lessons)
        sections.append("")

    # Instructions for the ReAct loop
    sections.append("---")
    sections.append("""## Begin Your Analysis

Use the Thought → Action → Observation loop.

**Thought:** What is happening? What is the key problem?
**Action:** query_memory | calculate_reward_budget | compare_rounds
**Observation:** Tool output

When you have enough information, output your FINAL ANSWER as a JSON proposal:

```json
{
    "diagnosis": "...",
    "root_cause_category": "overconstrained | underconstrained | reward_hacking | inactive_component | other",
    "changed_count": <1-3>,
    "proposed_changes": [...],
    "predicted_effect": "...",
    "max_risk": "...",
    "risk_mitigation": "..."
}
```

CRITICAL: changed_count must be ≤ 3. Prefer 1-2 changes over 3.
metrics_fn MUST be present in the generated code.
""")

    return "\n".join(sections)


class ReActLoop:
    """Simple ReAct loop that interleaves LLM calls with tool execution.

    The LLM generates Thought/Action lines. The loop parses actions,
    executes the corresponding tool, and feeds the observation back.
    """

    def __init__(self, system_prompt: str, memory_system: MemorySystem,
                 behavior_metrics: dict, component_means: dict,
                 reward_code: str, api_key: str, model: str = "deepseek-reasoner",
                 max_steps: int = 10):
        self.system_prompt = system_prompt
        self.memory = memory_system
        self.behavior_metrics = behavior_metrics
        self.component_means = component_means
        self.reward_code = reward_code
        self.api_key = api_key
        self.model = model
        self.max_steps = max_steps
        self.conversation_history = []

    def run(self, temperature: float = 0.4) -> tuple[str, list]:
        """Run the ReAct loop and return (final_proposal_text, conversation_history)."""
        for step in range(self.max_steps):
            # Build full conversation context into a single prompt
            parts = [self.system_prompt]
            for msg in self.conversation_history:
                role_prefix = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"\n\n{role_prefix}: {msg['content']}")
            parts.append("\n\nAssistant:")
            full_prompt = "".join(parts)

            response = call_llm(
                full_prompt,
                self.api_key, self.model, temperature,
            )

            self.conversation_history.append({"role": "assistant", "content": response})

            # Check for final JSON answer
            if "FINAL ANSWER" in response.upper() or "```json" in response:
                return self._extract_json(response), self.conversation_history

            # Parse action from response
            action = self._parse_action(response)
            if action is None:
                # No tool call — let the LLM continue thinking
                self.conversation_history.append({
                    "role": "user",
                    "content": "Continue. What do you observe? What is your next thought?"
                })
                continue

            # Execute tool
            observation = self._execute_tool(action["name"], action["input"])
            self.conversation_history.append({
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue your analysis."
            })

        # If we exhaust steps without a final answer, extract partial analysis
        last_response = ""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                last_response = msg["content"][:800]
                break

        # Extract any numerical observations from the conversation
        observations = {}
        if self.behavior_metrics:
            observations["behavior_metrics"] = self.behavior_metrics

        return json.dumps({
            "diagnosis": f"Analyst agent reached max steps ({self.max_steps}) without structured proposal. "
                         f"Partial analysis: {last_response}",
            "root_cause_category": "other",
            "changed_count": 1,
            "proposed_changes": [{
                "component": "general",
                "current_code": "See current reward function",
                "new_code": "Reward function unchanged (fallback)",
                "reason": f"Fallback: agent did not produce structured proposal. "
                          f"Behavior metrics: {observations.get('behavior_metrics', 'N/A')}"
            }],
            "predicted_effect": "No prediction available (max steps reached)",
            "max_risk": "Unknown — consider manual review",
            "risk_mitigation": "The reward function was NOT changed this round. "
                               "Check if a different diagnosis approach is needed.",
        }), self.conversation_history

    def _parse_action(self, text: str) -> Optional[dict]:
        """Parse tool calls from LLM response."""
        # Pattern: action_name: action_input
        for pattern in [
            r"query_memory:\s*(.+?)(?:\n|$)",
            r"calculate_reward_budget:\s*(.+?)(?:\n|$)",
            r"compare_rounds:\s*(.+?)(?:\n|$)",
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                action_type = pattern.split(":")[0].split("\\s")[0].lower()
                return {"name": action_type, "input": match.group(1).strip()}
        return None

    def _execute_tool(self, name: str, inp: str) -> str:
        """Execute a tool and return observation."""
        if name == "query_memory":
            lessons = self.memory.query_lessons(inp)
            if lessons:
                return "Relevant lessons:\n" + "\n---\n".join(lessons)
            return "No matching lessons found in memory."

        elif name == "calculate_reward_budget":
            try:
                budget = self.memory.calculate_reward_budget(
                    self.reward_code,
                    self.behavior_metrics,
                    self.component_means,
                )
                return json.dumps(budget, indent=2)
            except Exception as e:
                return f"Error calculating budget: {e}"

        elif name == "compare_rounds":
            try:
                nums = re.findall(r"\d+", inp)
                if len(nums) >= 2:
                    r1, r2 = int(nums[0]), int(nums[1])
                    rm1 = self.memory.get_round(r1)
                    rm2 = self.memory.get_round(r2)
                    ts1 = rm1.training_summary
                    ts2 = rm2.training_summary
                    return f"Round {r1}: {json.dumps(ts1.get('eval_history', [])[-1] if ts1 else {})}\nRound {r2}: {json.dumps(ts2.get('eval_history', [])[-1] if ts2 else {})}"
                return "Please specify two round numbers."
            except Exception as e:
                return f"Error comparing rounds: {e}"

        return f"Unknown tool: {name}"

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        # Try bare JSON object
        json_match = re.search(r"\{[\s\S]*\"diagnosis\"[\s\S]*\}", text)
        if json_match:
            return json_match.group(0)
        return text


def run_analyst_agent(run_dir: Path, round_num: int,
                       memory_system: MemorySystem,
                       api_key: str, model: str = "deepseek-reasoner",
                       temperature: float = 0.4) -> dict:
    """Run the analyst agent with ReAct loop.

    Returns:
        Structured proposal dict.
    """
    # Extract behavior metrics from perception report
    perception_path = run_dir / "perception_report.md"
    behavior_metrics = {}
    component_means = {}
    if perception_path.exists():
        report = perception_path.read_text("utf-8")
        _extract_metrics_from_report(report, behavior_metrics, component_means)

    # Load reward code
    reward_code = ""
    reward_path = run_dir / "reward_fn_source.py"
    if reward_path.exists():
        reward_code = reward_path.read_text("utf-8")

    # Build system prompt
    system_prompt = build_analyst_prompt(run_dir, round_num, memory_system)

    # Run ReAct loop
    loop = ReActLoop(
        system_prompt=system_prompt,
        memory_system=memory_system,
        behavior_metrics=behavior_metrics,
        component_means=component_means,
        reward_code=reward_code,
        api_key=api_key,
        model=model,
    )

    result_text, conversation_history = loop.run(temperature=temperature)

    # Save artifacts
    (run_dir / "analyst_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (run_dir / "analyst_conversation.json").write_text(
        json.dumps(conversation_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Parse or fallback
    try:
        proposal = json.loads(result_text)
    except json.JSONDecodeError:
        # Fallback: wrap text as a diagnostic
        proposal = {
            "diagnosis": result_text[:500],
            "root_cause_category": "other",
            "changed_count": 0,
            "proposed_changes": [],
            "predicted_effect": "See diagnosis",
            "max_risk": "Unknown",
            "risk_mitigation": "Monitor training metrics closely",
        }

    # Validate changed_count constraint
    if proposal.get("changed_count", 0) > 3:
        proposal["changed_count"] = len(proposal.get("proposed_changes", []))
        if proposal["changed_count"] > 3:
            proposal["proposed_changes"] = proposal["proposed_changes"][:3]
            proposal["changed_count"] = 3
            proposal["diagnosis"] += " [NOTE: truncated to 3 changes]"

    # Save proposal
    output_path = run_dir / "analyst_proposal.json"
    output_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")

    return proposal


def _extract_metrics_from_report(report: str, behavior: dict, components: dict):
    """Extract key numbers from perception report."""
    # Look for "Key Numbers for Budget Calculation" section
    section = re.search(
        r"Key Numbers.*?\n(.*?)(?=\n#)", report, re.DOTALL
    )
    if not section:
        section = re.search(
            r"Key Numbers.*?\n(.*?)$", report, re.DOTALL
        )

    if section:
        for line in section.group(1).split("\n"):
            if "mean_length" in line.lower():
                nums = re.findall(r"[-+]?\d*\.?\d+", line)
                if nums:
                    behavior["mean_length"] = float(nums[0])
            # Also extract any env-specific metrics (k=v pairs)
            # These are already handled via env_metadata in the analyst prompt
