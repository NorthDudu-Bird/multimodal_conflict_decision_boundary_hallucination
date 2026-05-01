# Results and Discussion Summary

在 300 张六色平衡评测集上，三模型在 `C0` 基线均达到 100% 视觉忠实，说明本任务在中性提示下不存在普遍性的基础识别失败。进入开放式图文冲突条件后，`Qwen2-VL-7B-Instruct` 与 `InternVL2-8B` 基本保持视觉一致；前者仅在 `C3/C4` 各出现 1 例 conflict-aligned 输出，后者在 `C0-C4` 中未出现 conflict-aligned 输出。相比之下，`LLaVA-1.5-7B` 在原始强误导模板下出现有限但可检测的偏移：`C3` 为 `27/300 = 9.00%`，`C4` 为 `10/300 = 3.33%`。

新增 paired 分析把这一结果显式化为 same-image answer flips。每个模型的 `C0-C4` 条件使用同一 300 张图，因此 LLaVA 在 `C3` 的 27 个 conflict-aligned 输出可表述为同图从 faithful C0 输出翻转到 false prompt color。这个表述比单独报告条件比例更强，因为它把变化绑定到相同视觉证据，而不是不同图像池之间的差异。

Prompt wording 边界控制同时限制了结论上限。将原始 `C3` 改写为两个语义接近的新 wording 后，`LLaVA-1.5-7B` 的 conflict-aligned rate 从 `9.00%` 降到 `C3-v2` 的 `1.67%`，并在 `C3-v3` 降为 `0.00%`；新 wording 下与稳定模型的差异不再显著。因此，更稳妥的表述不是“强误导提示下稳定出现语言偏差”，而是“在原始 C3/C4 模板下可观察到有限显著偏移，且该偏移对 wording 敏感”。

A1/A2 应作为辅助诊断阅读。它们说明 restricted answer space 和 counterfactual assumption 能诱发高 compliance，但不能替代 C0-C4 的主证据链。parser audit、source-stratified sanity check、reproducibility audit 和 completed visual clarity audit 则共同服务于 threats-to-validity，而不是新的主实验因素。

综合来看，当前证据支持一个窄而稳健的 empirical claim：在车身主颜色这一清晰单属性任务中，视觉证据总体占主导；`LLaVA-1.5-7B` 会在原始强误导开放式模板下表现出有限、显著、同图可配对追踪的 conflict-aligned shift；但该现象是模型依赖、条件依赖、模板依赖的局部行为，不能写成 VLM 普遍语言偏置或稳定跨模板规律。

## Phase 2 Discussion Addendum

The Phase 2 diagnostics make the paper's conclusion narrower and stronger. The main evidence chain remains C0-C4 plus same-image paired flips: visual evidence dominates overall, while LLaVA shows a limited and significant conflict-aligned shift under the original misleading templates. The new per-color split prevents an overbroad reading of that 9% result: it is concentrated mainly in `white->black`, not evenly spread across all colors.

The new prompt and format controls also show that the false-text effect is not a single universal property of misleading language. It depends on framing, correction affordance, and answer format. Some factorized prompts can produce larger conflict following than original C3, including title/prefix framing and no-correction presupposition, but quoted and indirect hints remain weak. Meanwhile, formal answer formats reduce the original LLaVA C3 effect. These findings should be written as attribution and boundary evidence, not as a prompt-engineering paper.

The completed visual audit and case taxonomy improve the Discussion by separating inspectable prompt-following flips, color-pair concentration, residual visual confounds, format sensitivity, source/style caveats, and multi-turn-induced failures. Multi-turn persuasion is especially important as a boundary condition: InternVL2 is stable in the original single-turn C0-C4 setup but becomes highly susceptible under repeated previous-turn false context. This belongs in the appendix or extended diagnostics because it tests a different interaction regime.
