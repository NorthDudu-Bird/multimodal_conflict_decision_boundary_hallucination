# Prompt Generation

当前主线没有单独维护一套独立的 prompt 生成脚本。

`strict-colors` 的 prompt 表生成已经并入：

- `scripts/data_prep/prepare_stanford_cars_multimodel_v2.py`

也就是说，数据筛选、最终 manifest 生成、annotation 表导出和 prompt CSV 生成是同一个当前主线入口，不需要再去找旧版单独 prompt 生成器。
