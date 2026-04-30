# Results and Discussion Summary

在 300 张六色平衡评测集上，三模型在 `C0` 基线均达到 100% 视觉忠实，说明本任务在中性提示下不存在普遍性的基础识别失败。进入开放式图文冲突条件后，`Qwen2-VL-7B-Instruct` 与 `InternVL2-8B` 基本保持视觉一致；前者仅在 `C3/C4` 各出现 1 例 conflict-aligned 输出，后者在 `C0-C4` 中未出现 conflict-aligned 输出。相比之下，`LLaVA-1.5-7B` 在原始强误导模板下出现有限但可检测的偏移：`C3` 为 `27/300 = 9.00%`，`C4` 为 `10/300 = 3.33%`。

新增 paired 分析把这一结果显式化为 same-image answer flips。每个模型的 `C0-C4` 条件使用同一 300 张图，因此 LLaVA 在 `C3` 的 27 个 conflict-aligned 输出可表述为同图从 faithful C0 输出翻转到 false prompt color。这个表述比单独报告条件比例更强，因为它把变化绑定到相同视觉证据，而不是不同图像池之间的差异。

Prompt wording 边界控制同时限制了结论上限。将原始 `C3` 改写为两个语义接近的新 wording 后，`LLaVA-1.5-7B` 的 conflict-aligned rate 从 `9.00%` 降到 `C3-v2` 的 `1.67%`，并在 `C3-v3` 降为 `0.00%`；新 wording 下与稳定模型的差异不再显著。因此，更稳妥的表述不是“强误导提示下稳定出现语言偏差”，而是“在原始 C3/C4 模板下可观察到有限显著偏移，且该偏移对 wording 敏感”。

A1/A2 应作为辅助诊断阅读。它们说明 restricted answer space 和 counterfactual assumption 能诱发高 compliance，但不能替代 C0-C4 的主证据链。parser audit、source-stratified sanity check、reproducibility audit 和 visual clarity audit infrastructure 则共同服务于 threats-to-validity，而不是新的主实验因素。

综合来看，当前证据支持一个窄而稳健的 empirical claim：在车身主颜色这一清晰单属性任务中，视觉证据总体占主导；`LLaVA-1.5-7B` 会在原始强误导开放式模板下表现出有限、显著、同图可配对追踪的 conflict-aligned shift；但该现象是模型依赖、条件依赖、模板依赖的局部行为，不能写成 VLM 普遍语言偏置或稳定跨模板规律。
