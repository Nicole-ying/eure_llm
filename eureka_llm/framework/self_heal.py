"""
self_heal.py — Fix a crashed reward function by calling the LLM with error context.

Called by pipeline.py when training exits non-zero.
The LLM receives the original prompt, the generated code, and the traceback,
then produces a fixed version of compute_reward.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

from llm_call import call_llm, extract_reward_fn

SELF_HEAL_PROMPT_TEMPLATE = """\
You previously generated a reward function for a reinforcement learning environment.
The function CRASHED during training with the error below.  Please fix it.

## Original prompt (provided when generating the reward function):
{original_prompt}

## Generated reward function (that crashed):
```python
{failing_code}
```

## Error traceback:
```
{traceback_text}
```

## Expected compute_reward signature
```python
def compute_reward(self, {expected_signature}):
```
The function signature above is what the environment actually calls.
Your fixed code MUST use this exact signature.

## Instructions
Fix the `compute_reward` function so it no longer crashes.
### Common crash causes (check these first):
1. **Storing simulation objects** — do NOT store simulator-internal objects
   (physics bodies, joints, handles) as instance attributes
   (e.g. `self.xxx = self.some_body.position`).  These cannot be pickled when using
   parallel environments.  Store only plain Python values (float, int, np.ndarray).
2. **Missing imports** — all imports must be inside the function body.
3. **Return type** — must return `(float_reward, dict_of_components)`.
4. **Wrong function signature** — the "Expected compute_reward signature" shows the
   exact argument list the environment passes.  Match it exactly.

Output your fixed `compute_reward` function in a single ```python block.
No explanation outside the block is needed.
"""


def build_fix_prompt(original_prompt: str, failing_code: str, traceback_text: str,
                     expected_signature: str = "action") -> str:
    template = SELF_HEAL_PROMPT_TEMPLATE
    for key, value in [
        ("original_prompt", original_prompt),
        ("failing_code", failing_code),
        ("traceback_text", traceback_text),
        ("expected_signature", expected_signature),
    ]:
        template = template.replace("{" + key + "}", value)
    return template


def validate_signature(code: str, expected_signature: str) -> str | None:
    """Check that compute_reward signature matches step.py's call pattern.

    Returns None if valid, or an error message string if mismatch found.
    """
    # Extract the function signature from generated code
    sig_match = re.search(r'def compute_reward\(self,\s*(.*?)\):', code)
    if not sig_match:
        return "Could not find compute_reward function signature in generated code"

    actual_sig = ' '.join(sig_match.group(1).split())
    expected = ' '.join(expected_signature.split())

    if actual_sig != expected:
        return f"Signature mismatch: expected ({expected}), but generated code has ({actual_sig})"

    return None


def adapt_and_verify_code(response_text: str) -> str:
    """Extract and verify the fixed reward code from LLM response."""
    if "```" not in response_text:
        # LLM might respond without code block — wrap the whole response
        # and try extracting anyway
        raise ValueError("No code block found in LLM response")
    code = extract_reward_fn(response_text)
    # Basic sanity checks
    if "compute_reward" not in code:
        raise ValueError("Fixed code missing compute_reward function")
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-prompt", required=True,
                        help="Path to the prompt file used in round0 or iteration")
    parser.add_argument("--failing-code", required=True,
                        help="Path to the reward_fn_source.py that crashed")
    parser.add_argument("--error-log", required=True,
                        help="Path to file containing stderr from the failed train run")
    parser.add_argument("--output-dir", required=True,
                        help="Where to save the fixed reward_fn_source.py")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (for API key)")
    parser.add_argument("--model", default="deepseek-reasoner",
                        help="LLM model for self-heal")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Lower temperature for more focused fix")
    args = parser.parse_args()

    original_prompt = Path(args.original_prompt).read_text(encoding="utf-8")
    failing_code = Path(args.failing_code).read_text(encoding="utf-8")
    traceback_text = Path(args.error_log).read_text(encoding="utf-8").strip()

    if not traceback_text:
        print("[self-heal] Error log is empty — nothing to fix.", file=sys.stderr)
        return 1

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    api_key = cfg.get("llm_api_key", os.environ.get("DEEPSEEK_API_KEY"))
    if not api_key:
        print("[self-heal] No API key found", file=sys.stderr)
        return 1

    prompt = build_fix_prompt(original_prompt, failing_code, traceback_text)
    print(f"[self-heal] Calling LLM ({args.model}) to fix reward function ...")

    response = call_llm(prompt, api_key, args.model, args.temperature)

    # Save artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "self_heal_prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "self_heal_response.md").write_text(response, encoding="utf-8")

    code = adapt_and_verify_code(response)
    code_path = output_dir / "reward_fn_source.py"
    header = f'"""LLM-generated reward function (self-healed).\nSource: {output_dir.name}\n"""\n\nimport math\nimport numpy as np\n\n'
    code_path.write_text(header + code + "\n", encoding="utf-8")
    print(f"[self-heal] Fixed code saved → {code_path}")

    # Print the fix summary for the pipeline log
    print(f"[self-heal] Reward function fixed and saved. Retrying training ...")


if __name__ == "__main__":
    sys.exit(main())
