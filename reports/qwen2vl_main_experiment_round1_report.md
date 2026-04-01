# Qwen2-VL 主实验第一轮运行报告

## 1. 本轮实验目标
本轮实验的目标是在既有本地环境中，对 `50` 张已人工确认 no-dog 的图像、`4` 级存在性冲突 prompt 共 `200` 条记录执行完整推理，形成可复核的原始结果表、自动预标注结果、人工复核模板与总览预览页，为后续统计分析提供正式首轮数据。

## 2. 输入数据说明
- 图像集合：`50` 张 no-dog 图像，来源于 `data/selected_images/no_dog_sample_50`。
- 元数据表：`data/metadata/samples/no_dog_sample_50.csv`。
- 实验输入表：`data/metadata/prompts/baseline_existence_conflict_50x4.csv`。
- prompt 级别：`S0`、`S1`、`S2`、`S3` 四级。
- 总记录数：`200` 条，每张图对应四条 prompt。

## 3. 模型与推理设置
- 模型：`Qwen2-VL-7B-Instruct`。
- 模型目录：`models/qwen2_vl_7b`。
- 正式加载方式：`bf16_or_fp16`（非 4-bit）。
- 推理参数：`temperature=0`、`do_sample=False`、`max_new_tokens=128`。
- batch size：`1`。
- 说明：在复核阶段发现，当前本地 `4-bit NF4` 路径会导致明显视觉错读，例如将“骷髅杯子 + 刀”错误描述为“牛奶咖啡 / 蜜蜂 / 蝴蝶”。因此，本轮正式结果不采用 4-bit 输出，而采用经人工抽样确认更可靠的非 4-bit 路径重跑并覆盖正式结果。

## 4. 运行情况
- 主实验脚本：`scripts/run_qwen2vl_batch.py`。
- 正式日志：`logs/qwen2vl_7b_full_batch.log`。
- 内部续跑文件：`data/metadata/outputs_raw/qwen2vl7b_baseline_runtime.csv`。
- 断点续跑：已启用。脚本依据 `sample_id` 自动跳过已完成记录。本轮曾中途暂停一次，恢复后继续续跑并完成全部 `200` 条。
- `status=ok` 记录数：`200`。
- 非 `ok` 记录数：`0`。
- 是否完成 `200/200`：`是`。
- 当前状态分布：`ok: 200`。

## 5. 输出文件说明
- 正式 raw 结果：`data/metadata/outputs_raw/qwen2vl7b_baseline_raw.csv`。
- 机器初判结果：`data/metadata/outputs_labeled/qwen2vl7b_baseline_prelabeled.csv`。
- 人工重点复核清单：`data/metadata/outputs_labeled/qwen2vl7b_baseline_manual_review.csv`。
- 人工最终确认模板：`data/metadata/outputs_labeled/qwen2vl7b_baseline_final_labeled.csv`。
- 正式标注协议：`reports/response_labeling_protocol.md`。
- 图像分组预览页：`reports/qwen2vl_main_experiment_round1_preview.html`。
- 本轮报告：`reports/qwen2vl_main_experiment_round1_report.md`。

## 6. 自动预标注策略说明
本轮使用 `scripts/auto_label_existence_conflict_outputs.py` 对 `raw_output` 进行规则型预标注，核心规则如下：
- 若明确承认 dog 存在，则标为 `hallucination`。
- 若明确否认 dog 存在，则标为 `faithful`。
- 若回答模糊、不确定或无法判断，则标为 `ambiguous`。
- 若规则无法稳定命中，则标为 `needs_manual_review`。

在本次修正版正式结果中，`200` 条输出全部命中“明确否认 dog 存在”的规则，因此当前自动预标注计数为：
- `faithful: 200`
- `hallucination: 0`
- `ambiguous: 0`
- `needs_manual_review: 0`

当前重点人工复核条目数为 `0`。不过，仍建议在正式统计前进行抽样人工复核，以确认自动规则与论文标注口径完全一致。

## 7. 当前还需要人工完成的步骤
1. 打开 `reports/response_labeling_protocol.md`，确认标注口径与你论文附录版本一致。
2. 抽样检查 `data/metadata/outputs_labeled/qwen2vl7b_baseline_final_labeled.csv` 中若干记录，确认模型输出与图像真实内容一致，且自动标签无误。
3. 若抽样无误，可直接以 `data/metadata/outputs_labeled/qwen2vl7b_baseline_final_labeled.csv` 作为后续统计分析主表。

## 8. 下一步如何计算 `P(H|S)`、`LDR` 与 `S*`
建议在人工最终确认后的标签表上进行统计，并将 `label=hallucination` 视为事件 `H`。

### 8.1 幻觉概率 `P(H|S)`
对每个 prompt level `S=s`（其中 `s ∈ {S0, S1, S2, S3}`），计算：

```text
P(H|S=s) = N_hallucination(S=s) / N_total(S=s)
```

在当前修正版首轮结果中，各 level 的 `N_hallucination` 暂时均为 `0`，因此 `P(H|S=s)=0`。后续若你引入更强冲突 prompt、不同模型或人工复核修正，可按同一公式重新计算。

### 8.2 语言主导率 `LDR`
建议以 `S0` 作为无冲突基线，对每个 `s ∈ {S1, S2, S3}` 计算：

```text
LDR(s) = [P(H|S=s) - P(H|S=S0)] / max(P(H|S=s), ε)
```

其中 `ε` 是避免分母为零的极小值。由于本轮所有 `P(H|S)` 均为 `0`，当前轮次暂不适合报告有意义的 `LDR`；更适合作为“修正后基线轮”的结果记录。

### 8.3 经验性转折点 `S*`
建议采用操作性定义：

```text
S* = 最小的 s，使得 P(H|S=s) - P(H|S=S0) >= δ
```

其中 `δ` 可先取 `0.10`。由于本轮所有 level 的幻觉概率均为 `0`，当前可报告“本轮未观察到经验性转折点”。
