# Run Diagnosis (BipedalWalker + LunarLander)

## Checked runs
- `runs/bipedalwalker-v3_2605061444_5000000`
- `runs/lunarlander-v2_2605061438_1000000`

## Primary failure pattern
1. `metrics_fn` often exists in generated code, but env-specific metrics are missing in eval summaries/history.
2. The framework previously swallowed runtime exceptions from `metrics_fn` in training wrappers, so failures were invisible.
3. Generator prompts were too broad and inconsistent on `metrics_fn` contract details, causing fragile code (signature drift, using `env.unwrapped`, engine-specific fields like `awake`).

## Why this causes the observed issue
- If `metrics_fn` throws at runtime, wrapper logic silently skipped metrics collection.
- Perception/analyst stages then see empty or sparse env metrics, reducing signal quality and making prompt iteration noisy.

## Applied framework changes
- Expose `metrics_fn` runtime errors in training trajectory logs (`metrics_fn_errors`).
- Expose `metrics_fn` runtime errors in behavior eval summary JSON.
- Tighten generator validation and prompt contract around:
  - exact `metrics_fn(env, action)` signature
  - no `env.unwrapped` usage in metrics function
  - avoid `awake/sleep` engine internals
  - scalar-only metric outputs

## Expected impact
- Faster diagnosis of bad generated reward modules.
- Better prompt focus (fewer ambiguous instructions).
- More stable cross-env behavior with lower prompt verbosity and stricter interface constraints.
