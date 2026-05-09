# Phase 4 Kickoff — HalfCheetah-v4 最小可运行闭环

## 本次已完成

1. 新增 `envs/HalfCheetah-v4/env.py`（带 `compute_reward` 钩子与 `reward_components` 输出）。
2. 新增 `envs/HalfCheetah-v4/step.py`（供 round0 prompt 读取 step 源码）。
3. 新增 `configs/ppo_1M_halfcheetah.yaml`（Phase-4 首个迁移配置样例）。

## 运行前依赖

HalfCheetah-v4 依赖 MuJoCo（Gymnasium mujoco extra）：

```bash
pip install "gymnasium[mujoco]"
```

## 建议最小流程

1. 先做环境探索，产出 `explorations/HalfCheetah-v4.json`（仓库已提供一个可用于 dry-run 链路验证的占位探索文件；真实实验建议重新采样生成）。
2. 用 round0 生成初始奖励。
3. 跑 1 轮迭代（dry-run 和真实训练各一次）。
4. 检查：
   - `*_guard.json`
   - `*_prompt_compaction.json`
   - `prompt_efficiency_report.md`
   - `constraints_report.json` / `critic_report.json`

## 下一步扩展

在 HalfCheetah 跑通后，按同一模板扩展到 Ant / Humanoid。

## 这个阶段是否会真的调用 LLM 生成奖励函数？

- `--dry-run`：**不会**调用 LLM，也不会训练；只生成/检查流程产物。
- `--mode full`（不加 `--dry-run`）：**会**调用 LLM（Round0/Analyst/Generator）并进入训练迭代。

可以直接用：

```bash
bash eureka_llm/tools/run_halfcheetah_phase4_step3.sh dry-run
# 或
DEEPSEEK_API_KEY=... bash eureka_llm/tools/run_halfcheetah_phase4_step3.sh train
```

## 一次性依赖安装脚本

- Linux: `bash eureka_llm/tools/setup_phase4_linux.sh`
- macOS: `bash eureka_llm/tools/setup_phase4_macos.sh`
