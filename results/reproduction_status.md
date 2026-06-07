# 复现执行状态

## 已完成

- 数据集重建：完成
- `C0` 全量重跑：完成
- `C0-C4` 全量重跑：完成
- `A1/A2` 全量重跑：完成
- `C3` prompt wording 鲁棒性控制全量重跑：完成
- parser audit 重建：完成
- Table 1 / Figure 2 / Table 3 / 附录图表重建：完成
- 主实验关键统计与附录 source sanity check 重建：完成
- 顶层文档与交付包整理：完成
- 复现比对：完成

## 复现结果

- 复现快照位置：`logs/reproducibility_snapshot/latest/`
- 比对文件：`results/reproducibility_comparison.csv`
- 审计摘要：`results/reproducibility_audit.md`
- 结论：本次全量重跑在 canonical 结果层面与锁定结果一致

## 当前可用结果目录

- `results/baseline/`
- `results/main/`
- `results/auxiliary/`
- `results/robustness/`
- `results/parser/`
- `results/appendix/`

## 说明

- 复现判定依据是 manifest、prompt、解析结果、条件统计、关键检验、parser audit、appendix sanity check 与总摘要等 canonical 论文资产。
- `*.log`、runtime CSV、run metadata JSON 与 raw free-text output 不属于阻断型复现门槛，只要解析结果和下游统计完全一致即可。
