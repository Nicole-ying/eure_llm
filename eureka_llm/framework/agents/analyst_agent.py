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
from prompt_guard import validate_zero_shot_output
from prompt_compaction import load_prompt_policy, summarize_structured_lines, write_compaction_stats
from prompt_harness import build_contract_block
from context_packet import build_evidence_packet

ANALYST_SYSTEM_PROMPT = (
    "You are the Analyst Agent. Use evidence to propose minimal, concrete reward-code changes."
)

def build_analyst_prompt(run_dir: Path, round_num: int,
                          memory_system: MemorySystem) -> tuple[str, dict]:
    """Build the full analyst prompt with context from multiple sources."""
    sys_prompt_path = Path(__file__).resolve().parent.parent.parent / "templates" / "analyst_system_prompt.txt"
    sys_prompt = sys_prompt_path.read_text("utf-8") if sys_prompt_path.exists() else ANALYST_SYSTEM_PROMPT

    # Load prompt policy for compaction limits
    compaction_stats = {}
    policy = load_prompt_policy(
        run_dir.parent if run_dir.name.startswith("round") else run_dir,
        "analyst",
    )

    # Load perception report
    perception_report = ""
    perception_path = run_dir / "perception_report.md"
    if perception_path.exists():
        perception_report, compaction_stats["perception_report"] = summarize_structured_lines(
            perception_path.read_text("utf-8"),
            max_lines=policy.get("max_lines_markdown", 80),
            keywords=("diagnosis", "principle", "mean", "std"),
        )

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
    lessons, compaction_stats["memory_lessons"] = summarize_structured_lines(
        memory_system.get_lessons(max_lines=120),
        max_lines=policy.get("max_lines_memory", 60),
        keywords=("round", "lesson", "why", "fix"),
    )
    recent_history = memory_system.get_recent_lessons(n=3)
    analyst_belief = memory_system.get_belief("analyst")

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

    critic_feedback_path = run_dir / "critic_feedback.json"
    if critic_feedback_path.exists():
        sections.append("### Critic/Constraints Feedback (requires revision-aware proposal)")
        sections.append("```json")
        summarized, st = summarize_structured_lines(critic_feedback_path.read_text("utf-8"), 50, ("critic", "principle", "risk"))
        compaction_stats["critic_feedback"] = st
        sections.append(summarized)
        sections.append("```")
        sections.append("")
    generator_feedback_path = run_dir / "generator_feedback.json"
    if generator_feedback_path.exists():
        sections.append("### Generator Feedback (translation/runtime failure context)")
        sections.append("```json")
        summarized, st = summarize_structured_lines(generator_feedback_path.read_text("utf-8"), 40, ("generator", "error", "failed"))
        compaction_stats["generator_feedback"] = st
        sections.append(summarized)
        sections.append("```")
        sections.append("")
    validation_issues = _load_generator_validation_issues(run_dir)
    if validation_issues:
        sections.append("### Must-Fix Generator Validation Issues")
        for i, issue in enumerate(validation_issues, 1):
            sections.append(f"{i}. {issue}")
        sections.append("")

    if recent_history:
        sections.append("### Recent Round History")
        sections.append(recent_history)
        sections.append("")

    actionable_lessons = []
    if lessons:
        sections.append("### Cross-Round Lessons (MEMORY.md)")
        sections.append(lessons)
        sections.append("")
        actionable_lessons = _extract_actionable_lessons(lessons)
        if actionable_lessons:
            sections.append("### Mandatory Lesson Carry-Over")
            for i, item in enumerate(actionable_lessons, 1):
                sections.append(f"{i}. {item}")
            sections.append("")
    if analyst_belief.get("history"):
        sections.append("### Analyst Persistent Belief State")
        sections.append("```json")
        sections.append(json.dumps({"recent": analyst_belief["history"][-5:]}, ensure_ascii=False, indent=2))
        sections.append("```")
        sections.append("")

    diagnostics = {}
    diag_path = run_dir / "perception_diagnostics.json"
    if diag_path.exists():
        try:
            diagnostics = json.loads(diag_path.read_text("utf-8"))
        except Exception:
            diagnostics = {}

    sections.append(build_evidence_packet(
        diagnostics=diagnostics,
        must_fix_issues=validation_issues,
        mandatory_lessons=actionable_lessons if lessons else [],
        max_items=5,
        strategy="strict",
    ))
    sections.append("")

    contract = build_contract_block(
        agent="Analyst",
        objective="Produce a minimal, high-confidence reward-change proposal grounded in evidence.",
        required_outputs=[
            "Valid JSON proposal with changed_count <= 3",
            "Concrete `new_code` for each proposed change",
            "Risk and mitigation tied to observed evidence",
        ],
        hard_constraints=[
            "If Must-Fix issues exist, each change must address at least one issue",
            "If Mandatory Lesson Carry-Over exists, proposal must not contradict it",
            "Prefer 1-2 focused changes over broad rewrites",
        ],
    )
    sections.append(contract)
    sections.append("")

    # Instructions for the ReAct loop
    sections.append("---")
    sections.append("""## Begin Your Analysis

Use the Thought → Action → Observation loop.
Focus discipline: identify ONE primary root cause first, then propose only the minimum edits needed.

**Thought:** What is happening? What is the key problem?
**Action:** query_memory | calculate_reward_budget | compare_rounds | analyze_efficiency | detect_principle_violation | ask_perception
**Observation:** Tool output

When you have enough information, output your FINAL ANSWER as a JSON proposal:

```json
{
    "diagnosis": "...",
    "violated_principle": "reward_goal_alignment | action_efficiency | exploration_balance | state_coverage | temporal_consistency | termination_exploitation | none",
    "root_cause_category": "overconstrained | underconstrained | reward_hacking | inactive_component | inefficiency | misalignment | exploration_collapse | other",
    "changed_count": <1-3>,
    "proposed_changes": [...],
    "predicted_effect": "...",
    "max_risk": "...",
    "risk_mitigation": "..."
}
```

CRITICAL: changed_count must be ≤ 3. Prefer 1-2 changes over 3.
metrics_fn MUST be present in the generated code.
If "Must-Fix Generator Validation Issues" exists, every proposed change MUST directly address at least one listed issue.
Use concrete `new_code` lines, not abstract principles only.
If "Mandatory Lesson Carry-Over" exists, do not contradict those items; reflect them in proposed_changes or risk_mitigation.
""")

    return "\n".join(sections), compaction_stats


def _extract_actionable_lessons(lessons_text: str) -> list[str]:
    lines = [ln.strip("- *") for ln in (lessons_text or "").splitlines() if ln.strip()]
    kws = ("remove", "delete", "must", "avoid", "terminal", "condition", "gate", "only if")
    picked = [ln for ln in lines if any(k in ln.lower() for k in kws)]
    return picked[:5]


def _load_generator_validation_issues(run_dir: Path) -> list[str]:
    path = run_dir / "generator_feedback.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception:
        return []
    issues = payload.get("validation_issues") or []
    return [str(x) for x in issues if str(x).strip()][:8]


class ReActLoop:
    """Simple ReAct loop that interleaves LLM calls with tool execution.

    The LLM generates Thought/Action lines. The loop parses actions,
    executes the corresponding tool, and feeds the observation back.
    """

    def __init__(self, system_prompt: str, memory_system: MemorySystem,
                 behavior_metrics: dict, component_means: dict,
                 reward_code: str, api_key: str, model: str = "deepseek-reasoner",
                 perception_query_fn=None, diagnostics: dict | None = None,
                 max_steps: int = 10):
        self.system_prompt = system_prompt
        self.memory = memory_system
        self.behavior_metrics = behavior_metrics
        self.component_means = component_means
        self.reward_code = reward_code
        self.api_key = api_key
        self.model = model
        self.perception_query_fn = perception_query_fn
        self.diagnostics = diagnostics or {}
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
            r"analyze_efficiency:\s*(.*?)(?:\n|$)",
            r"detect_principle_violation:\s*(.*?)(?:\n|$)",
            r"ask_perception:\s*(.*?)(?:\n|$)",
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

        elif name == "analyze_efficiency":
            action_mag = self.behavior_metrics.get("action_magnitude_mean") or self.diagnostics.get("action_magnitude_mean")
            velocity = self.behavior_metrics.get("velocity_mean") or self.diagnostics.get("velocity_mean")
            entropy = self.behavior_metrics.get("policy_entropy") or self.diagnostics.get("policy_entropy")
            notes = []
            if action_mag is not None and velocity is not None:
                ratio = velocity / max(abs(action_mag), 1e-6)
                notes.append(f"velocity/action_magnitude={ratio:.4f}")
                if abs(action_mag) > 0.9 and abs(ratio) < 0.3:
                    notes.append("High action amplitude with low movement gain: potential energy inefficiency.")
            if entropy is not None and entropy < 0.1:
                notes.append("Low policy entropy suggests potential exploration collapse.")
            if not notes:
                notes.append("Insufficient explicit efficiency metrics in behavior_metrics; use perception report cross-metric evidence.")
            return "\n".join(notes)

        elif name == "detect_principle_violation":
            violations = []
            mean_len = self.behavior_metrics.get("mean_length")
            if mean_len is not None and mean_len < 0.5:
                violations.append("termination_exploitation: episodes may be terminating too quickly relative to task horizon.")
            zero_std_components = [k for k, v in self.component_means.items() if isinstance(v, (int, float)) and abs(v) < 1e-9]
            if zero_std_components:
                violations.append(f"inactive_component: near-zero components detected: {zero_std_components[:3]}")
            for item in (self.diagnostics.get("constraint_violations") or [])[:3]:
                if isinstance(item, dict):
                    violations.append(
                        f"constraint_violation: {item.get('principle', 'unknown')} "
                        f"(severity={item.get('severity', 'n/a')})"
                    )
            if not violations:
                violations.append("No hard violation detected from structured diagnostics.")
            return "\n".join(violations)

        elif name == "ask_perception":
            if self.perception_query_fn is None:
                return "Perception follow-up unavailable in this run."
            try:
                return self.perception_query_fn(inp)
            except Exception as e:
                return f"Perception follow-up error: {e}"

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



def validate_proposal_focus(proposal: dict, must_fix_issues: list[str], mandatory_lessons: list[str]) -> list[str]:
    """Check proposal is concrete and aligned with required issues/lessons."""
    problems = []
    changes = proposal.get("proposed_changes") or []
    if not changes:
        problems.append("No proposed_changes provided.")
        return problems

    change_text = "\n".join(
        f"{c.get('component', '')}\n{c.get('new_code', '')}\n{c.get('reason', '')}"
        for c in changes if isinstance(c, dict)
    ).lower()

    for issue in must_fix_issues:
        tokens = [tok.lower() for tok in re.findall(r"[a-zA-Z_]{4,}", issue)[:4]]
        if tokens and not any(tok in change_text for tok in tokens):
            problems.append(f"Issue not addressed concretely: {issue}")

    for lesson in mandatory_lessons:
        lesson_tokens = [tok.lower() for tok in re.findall(r"[a-zA-Z_]{4,}", lesson)[:4]]
        if lesson_tokens and not any(tok in change_text for tok in lesson_tokens):
            problems.append(f"Mandatory lesson not reflected: {lesson}")

    for c in changes:
        if isinstance(c, dict) and not (c.get("new_code") or "").strip():
            problems.append(f"Change missing new_code: {c.get('component','unknown')}")
    return problems

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
    diagnostics = {}
    diag_path = run_dir / "perception_diagnostics.json"
    if diag_path.exists():
        try:
            diagnostics = json.loads(diag_path.read_text("utf-8"))
            for k, v in diagnostics.items():
                if isinstance(v, (int, float)):
                    behavior_metrics.setdefault(k, v)
        except Exception:
            diagnostics = {}
    if perception_path.exists():
        report = perception_path.read_text("utf-8")
        _extract_metrics_from_report(report, behavior_metrics, component_means)

    # Load reward code
    reward_code = ""
    reward_path = run_dir / "reward_fn_source.py"
    if reward_path.exists():
        reward_code = reward_path.read_text("utf-8")

    # Build system prompt
    system_prompt, compaction_stats = build_analyst_prompt(run_dir, round_num, memory_system)

    # Perception follow-up bridge (Phase-2: bidirectional communication)
    def _perception_query(question: str) -> str:
        from agents.perception_agent import answer_perception_query
        return answer_perception_query(run_dir, question)

    # Run ReAct loop
    loop = ReActLoop(
        system_prompt=system_prompt,
        memory_system=memory_system,
        behavior_metrics=behavior_metrics,
        component_means=component_means,
        reward_code=reward_code,
        api_key=api_key,
        model=model,
        perception_query_fn=_perception_query,
        diagnostics=diagnostics,
    )

    result_text, conversation_history = loop.run(temperature=temperature)

    # Save artifacts
    (run_dir / "analyst_prompt.txt").write_text(system_prompt, encoding="utf-8")
    write_compaction_stats(run_dir / "analyst_prompt_compaction.json", compaction_stats)
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

    must_fix_issues = _load_generator_validation_issues(run_dir)
    mandatory_lessons = _extract_actionable_lessons(memory_system.get_recent_lessons(6))
    focus_problems = validate_proposal_focus(proposal, must_fix_issues, mandatory_lessons)
    if focus_problems:
        repair_prompt = (
            system_prompt
            + "\n\n[REPAIR REQUIRED]\n"
            + "Your previous JSON did not satisfy required issue/lesson coverage.\n"
            + "Problems:\n- " + "\n- ".join(focus_problems[:8])
            + "\nReturn ONLY corrected JSON proposal with concrete new_code lines."
        )
        repaired_text = call_llm(repair_prompt, api_key, model, max(0.15, temperature - 0.2))
        try:
            proposal = json.loads(loop._extract_json(repaired_text))
            focus_problems = validate_proposal_focus(proposal, must_fix_issues, mandatory_lessons)
        except Exception:
            pass

    # Persist proposal focus guard for debugging
    (run_dir / "analyst_proposal_focus_guard.json").write_text(
        json.dumps({"passed": len(focus_problems) == 0, "problems": focus_problems[:12]}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

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
    (run_dir / "analyst_guard.json").write_text(
        json.dumps(validate_zero_shot_output(json.dumps(proposal, ensure_ascii=False)), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    memory_system.update_belief("analyst", {
        "round": round_num,
        "diagnosis": proposal.get("diagnosis", "")[:300],
        "violated_principle": proposal.get("violated_principle", "none"),
        "changed_count": proposal.get("changed_count", 0),
    })

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
