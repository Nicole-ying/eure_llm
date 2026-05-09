"""
constraint_discovery.py — Algorithmic (non-LLM) detection of transferable
training-dynamics constraint violations.
"""

from __future__ import annotations

from typing import Any


def detect_constraint_violations(traj_summary: dict, eval_history: list[dict]) -> list[dict[str, Any]]:
    """Return structured violations inferred from training dynamics only."""
    violations: list[dict[str, Any]] = []
    envm = traj_summary.get("env_metrics", {}) or {}
    comps = traj_summary.get("components", {}) or {}
    lengths = (traj_summary.get("lengths") or {})

    am = _mean_of(envm, "action_magnitude")
    vel = _mean_of(envm, "velocity")
    am_std = _std_of(envm, "action_magnitude")

    if am is not None and vel is not None and abs(am) > 1e-6:
        eff = vel / abs(am)
        if am > 0.9 and eff < 0.35:
            violations.append({
                "principle": "action_efficiency",
                "severity": "high",
                "evidence": {
                    "action_magnitude_mean": round(am, 4),
                    "velocity_mean": round(vel, 4),
                    "velocity_per_action": round(eff, 4),
                },
                "diagnosis": "High action amplitude but low movement gain; likely energy-inefficient behavior.",
            })
    if am_std is not None and am is not None and am_std < 0.1 * max(abs(am), 1e-6):
        violations.append({
            "principle": "action_efficiency",
            "severity": "medium",
            "evidence": {"action_magnitude_std": round(am_std, 4), "action_magnitude_mean": round(am, 4)},
            "diagnosis": "Action output variance is very low relative to its mean; policy may be saturating at a fixed action.",
        })

    for name, info in comps.items():
        cmean = float(info.get("mean", 0.0))
        cstd = float(info.get("std", 0.0))
        if abs(cmean) > 0.1 and cstd < 1e-6:
            violations.append({
                "principle": "reward_goal_alignment",
                "severity": "high",
                "evidence": {"component": name, "mean": round(cmean, 4), "std": round(cstd, 8)},
                "diagnosis": "Reward component has strong constant offset with near-zero variance; can be harvested without informative learning signal.",
            })

    if eval_history:
        mls = [float(r.get("mean_length", 0.0)) for r in eval_history if r.get("mean_length") is not None]
        if len(mls) >= 2 and max(mls) > 0:
            span = max(mls) - min(mls)
            if span / max(max(mls), 1e-6) < 0.1:
                violations.append({
                    "principle": "state_coverage",
                    "severity": "medium",
                    "evidence": {"mean_length_min": round(min(mls), 2), "mean_length_max": round(max(mls), 2)},
                    "diagnosis": "Episode lengths are concentrated in a narrow range; policy may be locked into a repetitive local optimum.",
                })
        # Temporal consistency proxy: compare early vs late evaluation windows.
        drift = _eval_window_drift(eval_history)
        if drift["max_relative_drift"] > 0.5:
            violations.append({
                "principle": "temporal_consistency",
                "severity": "medium",
                "evidence": {
                    "metric": drift["metric"],
                    "early_mean": round(drift["early"], 4),
                    "late_mean": round(drift["late"], 4),
                    "relative_drift": round(drift["max_relative_drift"], 4),
                },
                "diagnosis": "Behavior metric shifted strongly between early and late training windows; intra-policy dynamics may be inconsistent.",
            })

    lmin, lmax, lmean = lengths.get("min"), lengths.get("max"), lengths.get("mean")
    if isinstance(lmin, (int, float)) and isinstance(lmax, (int, float)) and isinstance(lmean, (int, float)) and lmax > 0:
        if lmean < 0.5 * lmax:
            violations.append({
                "principle": "termination_exploitation",
                "severity": "medium",
                "evidence": {"length_mean": round(lmean, 2), "length_max": round(lmax, 2), "ratio": round(lmean / lmax, 3)},
                "diagnosis": "Average episode length is far below observed max; agent may be exploiting early termination dynamics.",
            })

    return violations


def _mean_of(env_metrics: dict, key: str):
    v = env_metrics.get(key)
    return None if not isinstance(v, dict) else v.get("mean")


def _std_of(env_metrics: dict, key: str):
    v = env_metrics.get(key)
    return None if not isinstance(v, dict) else v.get("std")


def derive_action_cross_metrics(traj_summary: dict, eval_history: list[dict]) -> dict[str, Any]:
    """Derive action/behavior cross metrics for Phase-1 diagnostics."""
    envm = traj_summary.get("env_metrics", {}) or {}
    lengths = traj_summary.get("lengths", {}) or {}
    am = _mean_of(envm, "action_magnitude")
    vel = _mean_of(envm, "velocity")
    am_std = _std_of(envm, "action_magnitude")
    out: dict[str, Any] = {}
    if am is not None:
        out["action_magnitude_mean"] = round(float(am), 6)
    if am_std is not None:
        out["action_magnitude_std"] = round(float(am_std), 6)
    if am is not None and vel is not None and abs(am) > 1e-8:
        out["velocity_per_action"] = round(float(vel / abs(am)), 6)
    if eval_history:
        vals = [float(r.get("mean_length", 0.0)) for r in eval_history if r.get("mean_length") is not None]
        if vals:
            out["mean_length_span_ratio"] = round((max(vals) - min(vals)) / max(max(vals), 1e-8), 6)
    if {"mean", "max"} <= set(lengths.keys()) and lengths.get("max", 0):
        out["length_utilization_ratio"] = round(float(lengths.get("mean", 0.0)) / max(float(lengths["max"]), 1e-8), 6)
    return out


def derive_episode_consistency_metrics(traj_summary: dict, eval_history: list[dict]) -> dict[str, Any]:
    """Estimate within-episode behavioral consistency using temporal proxies.

    Since per-step full trajectories are not always persisted, this uses two robust proxies:
    1) early-vs-late metric drift in evaluation history
    2) action variability concentration from trajectory summary env metrics
    """
    envm = traj_summary.get("env_metrics", {}) or {}
    out: dict[str, Any] = {}
    drift = _eval_window_drift(eval_history)
    rel = float(drift.get("max_relative_drift", 0.0) or 0.0)
    out["early_late_relative_drift"] = round(rel, 6)
    out["early_late_consistency_score"] = round(max(0.0, 1.0 - min(rel, 1.0)), 6)
    if drift.get("metric") != "n/a":
        out["drift_dominant_metric"] = drift.get("metric")

    am = _mean_of(envm, "action_magnitude")
    am_std = _std_of(envm, "action_magnitude")
    if isinstance(am, (int, float)) and isinstance(am_std, (int, float)) and abs(am) > 1e-8:
        cv = abs(float(am_std)) / max(abs(float(am)), 1e-8)
        out["action_cv"] = round(cv, 6)
    return out


def _eval_window_drift(eval_history: list[dict]) -> dict[str, float | str]:
    if len(eval_history) < 4:
        return {"metric": "n/a", "early": 0.0, "late": 0.0, "max_relative_drift": 0.0}
    metric_series: dict[str, list[float]] = {}
    for row in eval_history:
        for k, v in (row.get("env_metrics") or {}).items():
            m = v.get("mean") if isinstance(v, dict) else None
            if isinstance(m, (int, float)):
                metric_series.setdefault(k, []).append(float(m))
    if not metric_series:
        return {"metric": "n/a", "early": 0.0, "late": 0.0, "max_relative_drift": 0.0}
    best = ("n/a", 0.0, 0.0, 0.0)
    for k, arr in metric_series.items():
        if len(arr) < 4:
            continue
        mid = len(arr) // 2
        early = sum(arr[:mid]) / max(mid, 1)
        late = sum(arr[mid:]) / max(len(arr) - mid, 1)
        rel = abs(late - early) / max(abs(early), 1e-6)
        if rel > best[3]:
            best = (k, early, late, rel)
    return {"metric": best[0], "early": best[1], "late": best[2], "max_relative_drift": best[3]}
