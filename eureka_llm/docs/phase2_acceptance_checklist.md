# Phase 2 验收规范（多智能体架构升级）

本清单用于判定 Phase 2 是否“可证据化完成”，避免仅凭主观判断。

## A. 事件驱动协调器

- [ ] 运行日志中出现事件回调输出：
  - `perception.completed`
  - `analyst.started`
  - `analyst.completed`
  - `constraints.completed`
  - `critic.completed`

## B. 持久信念状态

- [ ] 至少 4 类 agent belief 被写入（推荐：perception / analyst / generator / critic / constraints）。
- [ ] belief 中包含 round 或状态字段（如 `status`, `changed_count`, `count`）。

## C. 双向通信（Analyst ↔ Perception）

- [ ] `analyst_conversation.json` 中出现 `ask_perception:` 工具调用。
- [ ] 对应 observation 出现 `Perception follow-up` 返回内容。

## D. Critic/Constraints 反馈回路

- [ ] 存在 `constraints_report.json` 与 `critic_report.json`。
- [ ] 当 `critic_report.status == needs_revision` 时存在 `critic_feedback.json`。

## E. 交付物

- [ ] `tools/validate_phase2_completion.py` 验收结果为 PASS（或仅非关键 warning）。
- [ ] `tools/export_phase2_evidence.py` 成功导出 `phase2_evidence.csv`。

---

## 建议执行顺序

1. 先跑一次完整迭代（至少到 mid/late 角色阶段）。
2. 执行：
   - `python eureka_llm/tools/validate_phase2_completion.py --run-dir <run_dir>`
   - `python eureka_llm/tools/export_phase2_evidence.py --run-dir <run_dir>`
3. 将导出的 evidence 附到论文实验附录。
