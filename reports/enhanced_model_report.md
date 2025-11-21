# AAIP 增强版建模报告

**改进点**:
1. ✅ Bootstrap预测区间（95%置信区间）
2. ✅ 新增特征：节假日、Priority Sectors、累计配额、间隔异常
3. ✅ 分流超参数优化（大批量流vs小批量流）
4. ✅ 提高最小训练样本要求（8→12）

## 模型性能（滚动验证 MAE）

| Stream | Model | MAE | Test Size | Train Events |
| --- | --- | --- | --- | --- |
| Alberta Express Entry Stream | Linear | 113.5 | 16 | 28 |
| Alberta Express Entry Stream | RandomForest | 72.85 | 16 | 28 |
| Alberta Express Entry Stream | GradientBoosting | 82.85 | 16 | 28 |
| Dedicated Health Care Pathway | Linear | 25.76 | 6 | 18 |
| Dedicated Health Care Pathway | RandomForest | 37.99 | 6 | 18 |
| Dedicated Health Care Pathway | GradientBoosting | 39.54 | 6 | 18 |

## 最优模型

- **Alberta Express Entry Stream**: RandomForest
- **Dedicated Health Care Pathway**: Linear

## 预测结果（含95%置信区间）

| Stream | Date | Predicted | 95% CI Lower | 95% CI Upper | Model | Holiday |
| --- | --- | --- | --- | --- | --- | --- |
| Alberta Express Entry Stream | 2025-11-20 | 94 | 64 | 153 | RandomForest |  |
| Alberta Express Entry Stream | 2025-12-02 | 88 | 52 | 140 | RandomForest |  |
| Alberta Express Entry Stream | 2025-12-20 | 90 | 41 | 173 | RandomForest |  |
| Dedicated Health Care Pathway | 2025-10-19 | 71 | 0 | 225 | Linear |  |
| Dedicated Health Care Pathway | 2025-11-06 | 69 | 0 | 270 | Linear |  |
| Dedicated Health Care Pathway | 2025-12-03 | 38 | 0 | 339 | Linear |  |
