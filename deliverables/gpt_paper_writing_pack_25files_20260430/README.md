# 车身主颜色图文冲突实验主线

本仓库已经收敛为一条论文主线，只保留当前定版研究真正需要的内容：

- 平衡评测集：`data/balanced_eval_set/final_manifest.csv`
- `C0` 基线
- `C0-C4` 主实验
- `A1/A2` 辅助实验
- `C3` prompt wording 边界控制
- parser 映射审查
- 按 `source_dataset` 分层的附录 sanity check
- same-image paired flip 派生分析
- visual clarity / task-validity audit 基础设施

论文当前可保留的结论边界是：

- 三个模型在 `C0` 下都保持视觉忠实。
- 在原始强误导开放式模板下，`LLaVA-1.5-7B` 在 `C3` 和 `C4` 出现有限但显著的 conflict-aligned 行为。
- `Qwen2-VL-7B-Instruct` 与 `InternVL2-8B` 在当前任务中基本保持视觉一致。
- same-image paired 分析说明该现象可表述为同图从 faithful C0 输出翻转到 false prompt color 的有限样本级偏移。
- `C3` prompt wording 变体说明该现象对 wording 敏感，因此正文应写成“有限、模板敏感的语言偏差”，而不是稳定的跨模板规律。

## 先看这里

- [START_HERE.md](START_HERE.md)
- [docs/reproduction.md](docs/reproduction.md)
- [docs/experiment_plan.md](docs/experiment_plan.md)
- [docs/strengthening_master_plan.md](docs/strengthening_master_plan.md)
- [docs/project_audit.md](docs/project_audit.md)

## 官方工作流

以下 8 个入口构成唯一官方实验重建流程：

```bash
python scripts/build_dataset.py
python scripts/run_baseline_c0.py
python scripts/run_main_c0_c4.py
python scripts/run_aux_a1_a2.py
python scripts/run_robustness_c3_prompt_variants.py
python scripts/generate_parser_audit.py
python scripts/make_figures.py
python scripts/verify_reproducibility.py
```

以下脚本只做论文级派生分析和写作包整理，不触发模型推理：

```bash
python scripts/generate_paired_flip_analysis.py
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_visual_clarity_audit.py
python scripts/build_writing_pack.py --pack-date 20260430
```

## 关键输出

- 总结果摘要：[results/final_result_summary.md](results/final_result_summary.md)
- 主实验总表：[results/main/table1_main_metrics.md](results/main/table1_main_metrics.md)
- 主实验论文版解释：[results/main/main_results_paper_ready.md](results/main/main_results_paper_ready.md)
- same-image paired 分析：[results/main/paired_flip_summary.md](results/main/paired_flip_summary.md)
- 主实验关键统计：[results/main/main_stats_summary.md](results/main/main_stats_summary.md)
- 主实验关键检验：[results/main/main_key_tests.csv](results/main/main_key_tests.csv)
- 主图：[results/main/figure2_conflict_aligned_rates.png](results/main/figure2_conflict_aligned_rates.png)
- prompt 边界控制：[results/robustness/prompt_boundary_summary.md](results/robustness/prompt_boundary_summary.md)
- 辅助实验角色说明：[results/auxiliary/aux_role_note.md](results/auxiliary/aux_role_note.md)
- parser 审查：[results/parser/label_mapping_audit.md](results/parser/label_mapping_audit.md)
- 附录来源 sanity check：[results/appendix/stanford_core_sanity_check.md](results/appendix/stanford_core_sanity_check.md)
- visual clarity audit：[results/audit/visual_clarity_audit_readme.md](results/audit/visual_clarity_audit_readme.md)
- threats-to-validity 汇总：[results/threats_to_validity_summary.md](results/threats_to_validity_summary.md)
- 写作包升级说明：[docs/writing_pack_upgrade_note.md](docs/writing_pack_upgrade_note.md)

## 目录约定

- `data/` 与 `data_external/`：数据与来源元数据
- `prompts/c0_c4/`、`prompts/a1_a2/`：正式提示表
- `scripts/`：当前论文主线所需脚本和派生分析脚本
- `results/`：论文正式输出
- `docs/`：冻结计划、复现说明、补强计划和写作包说明
- `deliverables/`：最终 ZIP 交付包

## 本仓库已明确下线的内容

以下旧主线不再作为正文入口：

- 旧 `current` 配置树
- 旧 `current` prompt 树
- 旧分析目录
- 旧输出目录
- 旧预览与 review 目录
- 旧 Stanford-only 并列主线
- `decision boundary / threshold / LDI / RPE` 一类旧叙事包装

## 写作警戒线

- 不把当前结果写成“VLM 普遍存在语言偏置”。
- 不写成“文本普遍主导视觉”。
- 不把 A1/A2 当成主实验证据。
- 不把 C3 wording 变体写成鲁棒性成功；它的作用是限制主张边界。
- 不把 source/parser/reproducibility audit 写成新的主实验因素。

## 最终交付包

交付包索引见 [deliverables/README.md](deliverables/README.md)。
