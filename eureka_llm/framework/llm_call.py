"""
llm_call.py — LLM API call + code extraction.

Usage:
    python llm_call.py --prompt path/to/prompt.txt --output-dir path/to/round/
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def call_llm(prompt: str, api_key: str, model: str = "deepseek-reasoner",
             temperature: float = 0.6, timeout: float = 600.0) -> str:
    """Call DeepSeek API and return response text."""
    import httpx
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=httpx.Client(verify=False, follow_redirects=True, timeout=timeout),
    )
    print(f"  [LLM] Calling {model} (timeout={timeout}s, temp={temperature}) ...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        print(f"  [LLM] Response received ({len(response.choices[0].message.content)} chars)")
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM API call failed ({model}): {e}") from e


def extract_reward_fn(response_text: str) -> str:
    """Extract all relevant code blocks (compute_reward, metrics_fn, helpers)."""
    blocks = re.findall(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    combined = []
    for block in blocks:
        block = block.rstrip()
        if "def compute_reward" in block or "def metrics_fn" in block:
            combined.append(block)
    if not combined:
        raise ValueError(
            "No ```python block containing 'def compute_reward' found."
        )
    return "\n\n".join(combined) + "\n"


def extract_analysis(response_text: str) -> str:
    """Extract analysis text (everything outside code blocks)."""
    # Remove code blocks, return the rest
    cleaned = re.sub(r"```python\s*\n.*?```", "", response_text, flags=re.DOTALL)
    cleaned = re.sub(r"```\s*\n.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def save_artifacts(output_dir: Path, prompt: str, response: str,
                   analysis: str = None, code: str = None):
    """Save all LLM artifacts to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "response.md").write_text(response, encoding="utf-8")
    if analysis:
        (output_dir / "analysis.md").write_text(analysis, encoding="utf-8")
    if code:
        code_path = output_dir / "reward_fn_source.py"
        header = f'"""LLM-generated reward function.\nSource: {output_dir.name}\n"""\n\nimport math\nimport numpy as np\n\n'
        code_path.write_text(header + code + "\n", encoding="utf-8")
        print(f"  Code saved → {code_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (reads llm_api_key if --api-key not given)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and args.config:
        with open(args.config, encoding="utf-8") as f:
            try:
                import yaml  # type: ignore
                cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}
        api_key = cfg.get("llm_api_key")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set (provide via --api-key, --config, or env var)")
        sys.exit(1)

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    print(f"Calling LLM ({args.model}) ...")
    response = call_llm(prompt, api_key, args.model, args.temperature)
    print("Response received.")

    output_dir = Path(args.output_dir)
    save_artifacts(output_dir, prompt, response,
                   code=extract_reward_fn(response))
