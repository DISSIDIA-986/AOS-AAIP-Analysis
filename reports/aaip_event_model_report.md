# AAIP 抽签事件级建模报告（高级版）

数据源：阿尔伯塔省政府官网（processing page），2025 年抽签事件。

## 滚动验证 MAE（越低越好）

| Stream | Model | MAE | test_size | train_events |
| --- | --- | --- | --- | --- |
| Alberta Express Entry Stream | Linear | 143.82 | 20 | 28 |
| Alberta Express Entry Stream | RandomForest | 61.0 | 20 | 28 |
| Alberta Express Entry Stream | GradientBoosting | 62.98 | 20 | 28 |
| Alberta Opportunity Stream | Linear | 502.5 | 4 | 12 |
| Alberta Opportunity Stream | RandomForest | 164.66 | 4 | 12 |
| Alberta Opportunity Stream | GradientBoosting | 142.0 | 4 | 12 |
| Dedicated Health Care Pathway | Linear | 36.08 | 10 | 18 |
| Dedicated Health Care Pathway | RandomForest | 22.96 | 10 | 18 |
| Dedicated Health Care Pathway | GradientBoosting | 26.17 | 10 | 18 |


## 最优模型选择

- Alberta Express Entry Stream: RandomForest
- Alberta Opportunity Stream: GradientBoosting
- Dedicated Health Care Pathway: RandomForest


## 未来抽签事件预测

输出：`data/processed/aaip_event_forecasts_advanced.csv`

| Stream | projected_date | predicted_invitations | model | median_gap_days |
| --- | --- | --- | --- | --- |
| Alberta Express Entry Stream | 2025-11-20 00:00:00 | 87.4 | RandomForest | 6 |
| Alberta Express Entry Stream | 2025-12-02 00:00:00 | 80.5 | RandomForest | 6 |
| Alberta Express Entry Stream | 2025-12-20 00:00:00 | 82.9 | RandomForest | 6 |
| Alberta Opportunity Stream | 2025-12-02 00:00:00 | 936.4 | GradientBoosting | 22 |
| Alberta Opportunity Stream | 2026-01-15 00:00:00 | 517.6 | GradientBoosting | 22 |
| Alberta Opportunity Stream | 2026-03-22 00:00:00 | 517.2 | GradientBoosting | 22 |
| Dedicated Health Care Pathway | 2025-10-19 00:00:00 | 57.2 | RandomForest | 9 |
| Dedicated Health Care Pathway | 2025-11-06 00:00:00 | 59.5 | RandomForest | 9 |
| Dedicated Health Care Pathway | 2025-12-03 00:00:00 | 58.8 | RandomForest | 9 |
