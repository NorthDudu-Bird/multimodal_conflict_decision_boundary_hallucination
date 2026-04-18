# expanded_subset_build_log

## 流程
1. 审计 Stanford Cars 现有 strict clean manifest，并按最新 10 张人工歧义名单重建 `primary_core_stanford_only`。
2. 解包本地 `archive.zip`，生成 VCoR inventory。
3. 以 Stanford-only core 的缺口为目标，为六色建立 overfetch candidate pool。
4. 通过单车检测、主体占比、明暗、色彩匹配等规则做严格 auto screen。
5. 生成 `selected_manifest.csv` / `rejected_manifest.csv`。
6. 在 full rerun 后，对 repeated `other_wrong` 样本进行 spotcheck。
7. 发现 `vcor_train_black_08c4b7d380` 视觉上偏紫棕，执行人工 swap-out；改用 `vcor_test_black_dbdf4800f4` 替换，并重建 expanded manifest 与受影响推理结果。

## 关键数量
- stanford_only_total: 93
- expanded_balanced_total: 300
- selected_vcor_total: 207
- rejected_vcor_total: 1055
- target_per_color: 50
- manual_black_swap_count: 1

## VCoR 剔除主因
- secondary_car_interference: 426
- passes_auto_screen: 330
- highlight_overexposed: 65
- secondary_car_interference|highlight_overexposed: 55
- secondary_car_interference|weak_color_match: 34
- too_dark: 30
- weak_color_match: 28
- car_too_small: 23

## 复核状态
- 已做一次 targeted spotcheck。
- 1 张黑色样本因颜色边界不稳被替换。
- 目前仍保留一个保守的 optional human recheck 队列，见 `manual_recheck_queue.csv`。
