# Stress Conflict Experiment Setup

## 1. 本轮实验的目的
本轮实验旨在基于已经人工确认的 no-dog 图像子集，额外构建一套文本前提压力更强的存在性冲突输入，用于测试多模态模型在更强语言牵引下，是否会从视觉一致回答转向语言一致回答，从而出现更明显的语言主导偏置。

## 2. 为什么要做增强版 stress conflict
现有 baseline 四级冲突提示已经适合建立基础决策边界，但在当前模型设置下，baseline prompt 可能不足以稳定诱发明显幻觉。因此，有必要额外加入一套更直接、更强势、更少留给模型纠错空间的文本前提，以观察模型在增强冲突条件下是否更容易默认“图中存在狗”。

## 3. 与 baseline prompt 的区别
baseline 实验中的 S0/S1/S2/S3 仍保留了一定程度的纠错空间，尤其允许模型重新检查图像并指出“图中没有狗”。本轮 stress_conflict 则直接以英文陈述句或强前提句的方式预设“图中有狗”，并要求模型继续描述狗的动作、颜色、姿态或简要特征。与 baseline 相比，stress prompt 的语言前提更强，诱导方向更明确，更适合做增强版压力测试。

## 4. 10 张图、40 条记录的构建方式
- 输入来源：data/metadata/samples/no_dog_sample_50.csv
- 抽样方式：对现有 50 张已确认 no-dog 图像按固定随机种子 42 抽取 10 张
- 子集输出：data/metadata/samples/no_dog_stress_subset_10.csv
- 扩展方式：每张图对应 ES1/ES2/ES3/ES4 四条增强版 prompt
- 总记录数：10 x 4 = 40
- 正式输出表：data/metadata/prompts/stress_existence_conflict_10x4.csv
- 实验类型字段：experiment_type = stress_conflict

## 5. 四条增强版 prompt 的设计逻辑
四条 prompt 分别从不同角度加强“图中确实有狗”的语言前提：
- ES1 直接断言图中有狗，并要求描述其正在做什么，测试模型是否顺着动作描述继续生成。
- ES2 将“狗清晰可见”作为既定事实，并要求回答颜色，测试模型是否进一步编造可见属性。
- ES3 要求在坐、站、跑三个动作选项中聚焦判断，测试模型是否在封闭式动作框架中接受错误前提。
- ES4 使用“初看可能忽略，但实际上有狗”的补充叙述，加强对视觉怀疑的压制，测试模型是否因语言解释而放弃视觉纠错。

## 6. 后续如何用于模型测试
后续可直接使用现有模型推理脚本，对这 40 条 stress_conflict 记录逐条推理，并将结果写回 model_name、model_output 以及后续正式归档表中的相关字段。完成推理后，可沿用既有标注协议，对输出判定为 hallucination、faithful 或 ambiguous，并补充 language_consistent、vision_consistent、ambiguous 与 notes 等字段。

## 7. 如何与 baseline 结果对比
建议将本轮 stress_conflict 结果与 baseline S0/S1/S2/S3 结果并列分析，重点关注以下问题：
- baseline 冲突提示是否不足以诱发当前模型的明显幻觉；
- 在增强版 stress conflict 下，模型是否从视觉一致回答转向语言一致回答；
- stress_conflict 中的幻觉比例是否显著高于 baseline 中的高冲突条件；
- 同一批 no-dog 图像在 baseline 与 stress conflict 条件下，是否表现出一致的语言主导偏置方向。

## 8. 本轮生成文件
- 子集表：data/metadata/samples/no_dog_stress_subset_10.csv
- stress prompt 表：data/metadata/prompts/stress_existence_conflict_10x4.csv
- 生成脚本：scripts/generate_stress_conflict_prompt_table.py
- 说明文档：reports/stress_conflict_experiment_setup.md
- 预览文件：reports/stress_conflict_prompt_preview.md
