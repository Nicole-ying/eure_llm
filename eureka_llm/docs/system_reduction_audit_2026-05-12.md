# System Reduction Audit (2026-05-12)

## Scope Read
- `framework/pipeline.py`
- `framework/agents/{perception,analyst,generator,constraints,critic,reflection}_agent.py`
- `framework/{prompt_harness,context_packet,runtime_policy,prompt_compaction,memory/memory_system}.py`
- templates under `templates/*.txt`

## Prompt Review Criteria
1. 是否把注意力锁定到单一主因（single-root-cause discipline）。
2. 是否把“证据→改动”映射为可执行结构，而不是抽象标签。
3. 是否存在重复段落/空上下文块，导致 token 浪费和注意力稀释。
4. 是否有失败反馈的可操作性（actionable errors）。

## Main Findings
1. **Generator prompt had an empty evidence block** (no diagnostics/issues),增加 token 但无信息增益，属于纯噪声。
2. **Analyst evidence packet sourced from belief snapshot** instead of round-local diagnostics file,导致证据可能过时/缺失。
3. **Analyst instruction lacked explicit focus discipline**，容易输出多主题诊断（“都对一点”但都不够深）。
4. **Constraints/Critic remain lightweight rule-based guards**，当前适合作为“快速 fail-fast”，但不应伪装为深推理 agent。

## Changes Applied in This Iteration
1. Remove empty generator evidence packet to reduce prompt noise.
2. Analyst evidence packet now reads `perception_diagnostics.json` directly.
3. Add explicit “ONE primary root cause first” instruction in analyst ReAct section.

## Remaining Deletion Candidates
1. Collapse `constraints_agent` + `critic_agent` into single lightweight guard module unless LLM-critic is added.
2. Trim duplicated fallback text blocks if template files are guaranteed in production image.
3. Merge repeated JSON artifact writing patterns into utility helpers.

## Expected Impact
- 更少无效 token，提升提示聚焦度。
- Analyst proposal 与当前 round 证据绑定更紧，减少“用旧记忆修新问题”。
- 降低“多原因并列”导致的 generator 漂移风险。

