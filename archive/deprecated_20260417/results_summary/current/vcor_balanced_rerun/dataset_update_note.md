# dataset_update_note

- 正式保留两个数据版本：
  - `primary_core_stanford_only`
  - `primary_expanded_balanced_with_vcor`
- Stanford-only core 总数：93
- Expanded balanced 总数：300
- 新增 VCoR clean 图像：207
- VCoR 候选剔除：1055

## 6 色分布
- red: Stanford-only=40, Expanded=50, VCoR补充=10, 候选剔除=70
- blue: Stanford-only=12, Expanded=50, VCoR补充=38, 候选剔除=190
- green: Stanford-only=2, Expanded=50, VCoR补充=48, 候选剔除=240
- yellow: Stanford-only=2, Expanded=50, VCoR补充=48, 候选剔除=240
- black: Stanford-only=23, Expanded=50, VCoR补充=27, 候选剔除=135
- white: Stanford-only=14, Expanded=50, VCoR补充=36, 候选剔除=180

## 说明
- Stanford-only core 严格沿用最新 10 张人工歧义排除名单。
- Expanded 版本在不放宽 faithful 定义、不引入近似颜色判定的前提下，用 VCoR 只补 `red / blue / green / yellow / black / white` 六类缺口。
- 本轮最终达到每色 50 张，总计 300 张。
