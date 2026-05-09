#!/usr/bin/env python3
"""Generate real exploration JSON for HalfCheetah-v4 using env_explorer."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        default="eureka_llm/explorations/HalfCheetah-v4.json",
        help="Path to output exploration JSON",
    )
    args = ap.parse_args()

    try:
        from eureka_llm.framework.env_explorer import explore_from_gym
    except Exception as e:
        raise RuntimeError(
            "Failed to import env_explorer dependencies. "
            "Please install required packages (numpy, gymnasium[mujoco])."
        ) from e

    data = explore_from_gym("HalfCheetah-v4", n_episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote exploration -> {out}")


if __name__ == "__main__":
    main()
