# Dataset Preparation Report

## 1. 数据来源
- 数据集：COCO 2017 `val2017`
- 图像下载地址：`http://images.cocodataset.org/zips/val2017.zip`
- 标注下载地址：`http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
- 处理时间：`2026-03-31 09:19:17 UTC`

## 2. 下载文件信息
- `data/raw/val2017.zip`：size=777.80 MB
- `data/raw/annotations_trainval2017.zip`：size=241.19 MB
- 解压图像目录：`data/coco/val2017/`
- 解压标注目录：`data/coco/annotations/`

## 3. dog 的 category_id
- `dog` 的 category_id = **18**

## 4. 含 dog 图片数量
- 含 `dog` 标注的图片数量：**177**

## 5. 不含 dog 图片数量
- 不含 `dog` 标注的图片数量：**4823**
- 完整清单：`data/metadata/no_dog_all.csv`

## 6. 经过轻量筛选后的候选数量
- 轻量筛选条件：
  - 宽度 >= 300
  - 高度 >= 300
  - `num_annotations` <= 15
- 通过轻量筛选的候选数量：**4093**
- 候选清单：`data/metadata/no_dog_filtered_candidates.csv`

## 7. 最终抽样 50 张的说明
- 抽样来源：轻量筛选后的候选池
- 抽样数量：**50**
- 随机种子：**42**
- 样本元数据：`data/metadata/no_dog_sample_50.csv`
- 样本图片目录：`data/selected_images/no_dog_sample_50/`
- HTML 预览页：`reports/no_dog_sample_50_preview.html`
- Contact sheet：`data/previews/no_dog_sample_50_contact_sheet.jpg`

## 8. 当前自动筛选的局限性
- 当前仅做了基础存在性过滤：通过 COCO 标注判断图片中是否存在 `dog`。
- 当前仅做了轻量质量控制：尺寸阈值和标注数量阈值。
- 未自动检测模糊、遮挡、极端裁切、强反光、文字干扰、复杂拥挤背景等情况。
- COCO 标注并不保证“视觉上绝对不存在狗”，只保证“无 `dog` 类别标注”，因此仍需人工二次复核。
- 已生成 contact sheet，便于人工快速目检。

## 9. 建议的人工复核步骤
1. 打开 `reports/no_dog_sample_50_preview.html`，逐张确认图像中确实不存在狗。
2. 优先剔除背景过于复杂、主体不清晰、难以构造稳定冲突提示的图像。
3. 在 `data/metadata/no_dog_sample_50.csv` 的 `notes` 列中记录保留/剔除理由。
4. 若发现隐藏狗、疑似狗、玩具狗、卡通狗或局部狗元素，建议直接剔除。
5. 完成人工复核后，冻结一版“最终实验样本清单”，避免后续实验中样本漂移。
