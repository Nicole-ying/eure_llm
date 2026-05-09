# 周报 2026-04-24

---

## 本周工作概述

本周推进了两件事：

1. **迭代闭环系统调试完成**：`run_5_rounds.py` 全自动 5 轮迭代管线已跑通，5M 实验当前运行中（Round 0 即将完成）
2. **近期文献调研**：梳理了 2025–2026 年与 LLM reward design 高度相关的 15+ 篇工作，形成下一阶段系统改进方向

---

## 一、框架推进状态

### 核心进展

| 组件 | 状态 |
|------|------|
| 两阶段 LLM 迭代（分析→生成） | ✅ 完成 |
| `run_5_rounds.py` 自动管线 | ✅ 完成 |
| 实验产物按规范目录组织 | ✅ 完成 |
| 奖励函数自动验证层 | ✅ 完成 |
| OMP 死锁修复 / SubprocVecEnv | ✅ 完成 |
| GIF / checkpoint / evaluation 输出 | ✅ 完成 |
| 支持 `--resume` 断点续训 | ✅ 完成 |
| **5 轮全自动实验** | 🔄 运行中（5M 步/轮 × 5 轮） |

### 当前实验

配置 `ppo_5M_cpu.yaml`（n_envs=4, 5M steps/round），Round 0 已完成 ~85%，预计本日完成全部 5 轮迭代。

---

## 二、文献调研与系统改进方向

搜索了 2025–2026 年与 LLM reward design 最相关的工作，以下是可以融入系统架构的具体方向，排名按实施优先级从高到低：

### 方向 1：轻量级快速验证（Fast Rejection）

**问题**：当前每轮训练完整 5M 步（~107 min），如果奖励函数无效（分量失衡、梯度消失、reward hacking），全部浪费。

**借鉴工作**：
- **CARD** (Knowledge-Based Systems 2025) 的 TPE 机制——不每次跑完整 RL 训练，用少量 trajectory preference 评估奖励质量

**具体方案**：
- 每轮新增 Fast Verification 阶段：用新奖励函数训练 50K 步 → 检查奖励分量量级是否平衡、行为指标是否有改善趋势
- 如果验证未通过，立即拒绝并让 LLM 重新生成，不再跑完 5M
- 预计节省 90%+ 的无效训练时间

### 方向 2：奖励函数种群进化（Population Evolution）

**问题**：当前每轮只生成一个奖励函数，没有候选机制，"坏"的一轮浪费一次迭代机会。

**借鉴工作**：
- **R\*** (ICML 2025, Li et al.)：维护 reward function population，用 LLM 做 mutation operator + module-level crossover，用多个 critic 投票选择高质量 trajectory 做偏好学习
- **REvolve** (ICLR 2025, Hazra et al.)：演化式框架，将人类隐式知识翻译为显式奖励函数

**具体方案**：
- 每轮让 LLM 生成 3 个候选变体（通过温度采样或 prompt 变体）
- 每个候选跑轻量级快速验证（方向 1）
- 用 behavioral metrics 排序，选最优的进入下一轮
- 保留失败候选的分析结论，注入下一轮 prompt

### 方向 3：奖励结构 / 参数解耦（Structure-Parameter Decoupling）

**问题**：当前 LLM 同时决定奖励函数的"结构"（用什么项）和"参数"（系数值），搜索空间过大。

**借鉴工作**：
- **R\*** (ICML 2025)：明确分为 reward structure evolution + parameter alignment optimization 两步
- **Text2Reward** (ICLR 2024)：模板式生成，结构由模板定义，LLM 只需填充参数

**具体方案**：
- Step 1：LLM 只决定奖励结构——有哪些分量、每分量的数学形式
- Step 2：固定结构后，用小型参数搜索（grid search 或 LLM 建议的 range）优化系数
- Step 3：用最优参数跑完整训练
- 降低 LLM 单次决策难度，提高成功率

### 方向 4：多 LLM 独立分析 + 综合（Multi-Agent Analysis）

**问题**：当前分析阶段只调用一次 LLM，诊断质量受单次生成质量波动影响。

**借鉴工作**：
- **CVCP** (Symmetry 2025)：multi-agent cross-verification protocol，多个 agent 独立分析后投票/综合
- **AgentForge** (arXiv 2026)：Planner → Coder → Tester → Debugger → Critic 的 pipeline

**具体方案**：
- 两个独立 LLM 调用，分析同一份训练数据
- 第三个 LLM 作为 meta-agent，综合两份分析报告
- 解决分歧、确认共识、生成更稳健的改进建议
- 这是你提到的创新点二（多 Agent 系统）的具体实现入口

### 方向 5：训练中 Reward Hacking 检测（Runtime Monitoring）

**问题**：当前没有任何机制在训练过程中检测 reward hacking——agent 可能在"骗"奖励函数而非真正完成任务。

**借鉴工作**：
- **TRACE** (ICLR 2026 Oral)：通过测量 reasoning effort 检测 reward hacking，核心洞察是"hacking 比真正解决问题更容易"
- **Cooper** (2025)：联合优化 policy 和 reward model，检测 reward model 被利用的模式
- **Specification Self-Correction** (2025)：模型在 test-time 自我识别奖赏规格漏洞

**具体方案**：
- 训练中持续对比 reward 趋势和 behavioral metrics 趋势
- 如果 reward 持续上升但 completion_rate / fall_rate 等行为指标不改善 → 触发 hacking 警报
- 记录 hacking episode 的 obs-action 序列，作为下一轮分析的补充材料
- 让 LLM 在下一轮生成时"修补漏洞"

### 方向 6：行为轨迹偏好回放（Trajectory Preference）

**问题**：当前传给 LLM 的是聚合统计（均值、std），丢弃了具体轨迹信息。

**借鉴工作**：
- **CARD**：TPE 机制——直接对比"成功轨迹"和"失败轨迹"上的奖励分布差异

**具体方案**：
- 挑选典型 episode 的完整轨迹（obs/action/reward_component 序列）
- 成功案例：最终走完全程的 / 高 completion_rate 的
- 失败案例：早期摔倒的 / 低 completion_rate 的
- 让 LLM 对比两条轨迹上的奖励分量差异，定位问题
- 这个方向与你已有的奖励分量轨迹日志（trajectory_logs/*.jsonl）完全兼容，数据结构已经就绪

### 方向 7：自验证 + 预测反馈（Self-Verification Loop）

**问题**：LLM 生成奖励函数后，要到训练结束才知道效果，反馈周期太长。

**借鉴工作**：
- **ReVeal** (2025)：interleave code generation with explicit self-verification，模型生成测试用例并在沙箱中执行验证
- **Absolute Zero Reasoner** (NeurIPS 2025)：模型自己提议任务、自己解决、自己的学习进度作为奖励信号

**具体方案**：
- 在 LLM 生成奖励函数时，额外要求输出 behavioral prediction："预计本次改进会使 completion_rate 提升 X%，mean_length 变化 Y"
- 训练后将预测与实际对比
- 预测误差作为 meta 信号，用于评估 LLM 的"自我认知能力"
- 长期来看，可以让 LLM 学会更准确地预测自己的设计效果

### 汇总对比

| 方向 | 主要借鉴 | 收益 | 实施成本 | 优先级 |
|------|---------|------|---------|-------|
| ① 快速验证 | CARD | 节省 90%+ 无效训练时间 | 低 | ★★★★★ |
| ② 种群进化 | R\*, REvolve | 提高每轮成功率 | 中 | ★★★★ |
| ③ 结构/参数解耦 | R\*, Text2Reward | 降低 LLM 决策难度 | 中 | ★★★★ |
| ④ 多 LLM 分析 | CVCP, AgentForge | 提高诊断鲁棒性 | 低 | ★★★★ |
| ⑤ Hacking 检测 | TRACE, Cooper | 避免虚假改进 | 低 | ★★★ |
| ⑥ 轨迹回放 | CARD | 信息更丰富的反馈 | 中（数据已就绪） | ★★★ |
| ⑦ 自验证循环 | ReVeal, AZR | 缩短反馈周期 | 高 | ★★ |

### 推荐路线

最近一期：**① 快速验证 + ④ 多 LLM 分析**（实施成本最低，收益最明确）
- 快速验证：在现有 validation 层基础上，加一个 50K 步的快速训练检查
- 多 LLM 分析：分析阶段改为 2 个独立 LLM + 1 个 meta-agent

下一期：**② 种群进化 + ③ 结构/参数解耦**
- 这两者天然互补，R\* 已经证明了组合效果

远期：**⑤ Hacking 检测 + ⑥ 轨迹回放 + ⑦ 自验证**
- 数据基础设施已经具备（trajectory_logs、evaluation history），主要是 prompt 设计

---

## 三、论文引用汇总

具体引用和借鉴说明见附录。核心参考文献及其 2025–2026 会议/期刊分布：

| 论文 | 会议/期刊 | 年份 | 本系统的借鉴点 |
|------|----------|------|-------------|
| Eureka | ICLR 2024 | 2024 | 基线对比，fitness score 依赖的替代方案 |
| CARD | Knowledge-Based Systems | 2025 | TPE 轨迹偏好评估，免每轮 RL 训练 |
| R\* | ICML 2025 | 2025 | 奖励结构演化 + 参数对齐优化 |
| REvolve | ICLR 2025 | 2025 | 演化式奖励函数优化 |
| TRACE | ICLR 2026 Oral | 2025 | Reward hacking 检测 |
| Cooper | arXiv | 2025 | 联合优化 + hacking 模式检测 |
| ReVeal | arXiv | 2025 | 自验证生成循环 |
| CVCP | Symmetry | 2025 | 多 agent 交叉验证协议 |
| AgentForge | arXiv | 2026 | 执行沙箱验证 + 多角色 agent pipeline |
| Absolute Zero | NeurIPS 2025 | 2025 | 自提议+自解决的闭环学习 |
| Text2Reward | ICLR 2024 | 2024 | 模板化奖励设计 |

---

下一步：等待本轮 5 轮实验完成，根据实验结果确定方向 ①（快速验证）的具体阈值设计，然后进入实现阶段。
