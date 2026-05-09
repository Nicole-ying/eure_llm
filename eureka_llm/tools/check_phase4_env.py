#!/usr/bin/env python3
"""Check local dependencies for Phase-4 HalfCheetah pipeline and print install hints."""

from __future__ import annotations
import importlib
import os
import subprocess
import sys


def check_pkg(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def main():
    checks = [
        ("numpy", "pip install numpy"),
        ("gymnasium", "pip install gymnasium"),
        ("openai", "pip install openai"),
    ]
    results = []
    for mod, hint in checks:
        ok, msg = check_pkg(mod)
        results.append((mod, ok, msg, hint))

    mujoco_ok = False
    mujoco_msg = "unknown"
    try:
        import gymnasium as gym
        env = gym.make("HalfCheetah-v4")
        env.reset(seed=0)
        env.close()
        mujoco_ok = True
        mujoco_msg = "ok"
    except Exception as e:
        mujoco_msg = str(e)

    print("=== Phase4 Environment Check ===")
    for mod, ok, msg, hint in results:
        print(f"- {mod}: {'OK' if ok else 'MISSING'}")
        if not ok:
            print(f"  reason: {msg}")
            print(f"  install: {hint}")

    print(f"- HalfCheetah-v4 runtime: {'OK' if mujoco_ok else 'MISSING'}")
    if not mujoco_ok:
        print(f"  reason: {mujoco_msg}")
        print("  install: pip install \"gymnasium[mujoco]\"")

    key = os.environ.get("DEEPSEEK_API_KEY")
    print(f"- DEEPSEEK_API_KEY: {'SET' if key else 'NOT SET'}")
    if not key:
        print("  export DEEPSEEK_API_KEY=your_key")

    all_ok = all(ok for _, ok, _, _ in results) and mujoco_ok and bool(key)
    print(f"\nSummary: {'READY' if all_ok else 'NOT READY'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
