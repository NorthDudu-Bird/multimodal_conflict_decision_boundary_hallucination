# 复现执行状态

## 已完成

- 数据集主线重构：完成
- 平衡评测集 metadata 重建：完成
- `C0` 基线路径全量 rerun：完成
- `C0-C4` 主实验新目录全量 raw rerun：完成
- `A1/A2` 辅助实验新目录全量 raw rerun：完成
- `analyze_results.py` 聚合分析重算：完成
- Table 1 / Figure 2 / Table 3 / 数据集分布图：完成
- 文档重写：完成
- 旧主流程归档：完成

## 当前状态

当前论文主线所需的四类结果目录已经齐备：

- `results/baseline/`
- `results/main/`
- `results/auxiliary/`
- `results/appendix/`

其中 `main` 与 `auxiliary` 的逐样本 raw 结果、解析结果、聚合统计、代表性案例和图表都已经与本次重构后的脚本入口对齐。

## 备注

- 为避免 Windows 保留名 `AUX` 带来的路径兼容性问题，辅助实验的论文输出目录已统一规范为 `results/auxiliary/`。
- 历史生成的 `results/aux/` 仅保留为兼容过渡痕迹，不再作为默认引用路径。
