# Framework Audit vs Research Goal (2026-05-08)

## Goal being audited
"Multi-agent reward auto-design without official reward; iterate from training dynamics only."

## Status Summary
- ✅ Multi-agent pipeline exists (Perception → Analyst → Generator → Reflection).
- ✅ No official reward is required for reward-function iteration.
- ✅ Internal signals are collected: behavior metrics, component attribution, env metrics, policy entropy.
- ⚠️ Not yet a true search framework: currently single-path iterative update, no population manager.
- ⚠️ No MCTS / Thompson / novelty maintenance implementation yet.
- ⚠️ No unified scalar objective (TDRQ) in prior version; now added as first integrated index for perception-time diagnosis.

## What is implemented now
1. Training-dynamics capture
   - reward component means/stds (CARD-style episode summaries)
   - behavior eval metrics (completion/fall/truncation/length)
   - env-specific metrics via `metrics_fn`
   - policy entropy history (`entropy_history.jsonl`)

2. Multi-agent iteration loop
   - Perception report generation from logs and trajectory summaries
   - Analyst proposal JSON generation
   - Generator code synthesis + validation
   - Reflection and memory update

3. Reliability additions
   - metrics_fn runtime errors surfaced to logs/eval summaries
   - stricter generator contract for metrics function robustness

## Missing for your paper's full claim
1. Population/search layer (critical)
   - missing: candidate pool, branching, elite selection, diversity control
   - missing: search controller (MCTS/UCB/Thompson or similar)

2. Explicit optimization target across rounds
   - before this patch: no integrated TDRQ scalar
   - now: TDRQ index added for perception diagnosis, but not yet used as automatic selection objective in branching search

3. Controlled ablation protocol
   - no built-in experiment harness for "with/without official reward" or "single-path vs multi-path"

## Immediate next engineering steps
1. Add `search_manager.py` with candidate registry and branching budget per round.
2. Use TDRQ as one objective in candidate ranking (with behavior safety gates).
3. Add novelty/diversity score over reward component fingerprints.
4. Add experiment launcher for ablations and auto-table export.
