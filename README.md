# eureka_llm

## Phase-3 Prompt Policy (示例)

你可以在每个 run 的 `config.yaml` 里配置统一的 prompt 压缩策略：

```yaml
phase3:
  prompt_policy:
    perception:
      max_env_metrics_table: 6
      max_env_metrics_section: 6
    analyst:
      max_lines_markdown: 90
      max_lines_memory: 70
    generator:
      max_lines_markdown: 85
```

说明：
- `perception`：控制 `template_engine` 中环境指标筛选数量。
- `analyst`：控制 perception report / memory / feedback 摘要保留行数。
- `generator`：控制 task manifest / perception report 的摘要保留行数（奖励代码本体完整保留）。

也可直接复制模板：`eureka_llm/configs/phase3_prompt_policy.yaml`。

## Prompt Compaction 统计汇总脚本

新增脚本：`eureka_llm/tools/summarize_prompt_compaction.py`

用途：
- 扫描 `runs/*/round*/` 下的 `*_prompt_compaction.json`
- 汇总每轮 keep/drop 比例
- 导出 CSV，便于论文画图

示例：

```bash
python eureka_llm/tools/summarize_prompt_compaction.py \
  --runs-root eureka_llm/runs \
  --output eureka_llm/runs/prompt_compaction_summary.csv
```
