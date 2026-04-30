# Deliverables

当前目录保留论文写作和检查用交付包。

## 推荐使用

- `gpt_paper_writing_pack_20files_20260430.zip`
  - 面向“最多只能上传 20 个文件”的论文写作场景。
  - 当前最推荐上传给 GPT 的压缩包。
  - 保留所有关键证据类别：主结果、paired flip、prompt boundary、A1/A2 角色、parser/source/reproducibility/visual clarity threats。
- `gpt_paper_writing_pack_25files_20260430.zip`
  - 面向“最多只能上传 25 个文件”的论文写作场景。
  - 新增 paired flip、prompt boundary、auxiliary role、threats-to-validity 和 visual clarity audit 说明。
  - 如果平台允许 25 个文件，可使用该包获取更多表图文件。

## 历史包

- `gpt_paper_writing_pack_20260418.zip`
  - 早期论文整理与归档包。
- `gpt_experiment_check_pack_20260418.zip`
  - 早期实验检查包。

## 写作边界

使用交付包时必须保持以下边界：

- 不改研究问题。
- 不新增模型或任务。
- 不把局部结果写成 VLM 普遍语言偏置。
- 不把 C3 wording boundary 写成跨模板鲁棒性成功。
- 不把 A1/A2 当成 C0-C4 主证据。

如果需要重新生成 20260430 写作包，请运行：

```bash
python scripts/build_writing_pack.py --pack-date 20260430 --file-cap 20
python scripts/build_writing_pack.py --pack-date 20260430 --file-cap 25
```
