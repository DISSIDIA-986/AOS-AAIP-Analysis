# AAIP历史数据分析与预测（2025 抽签密集版）

从阿尔伯塔省政府官网 https://www.alberta.ca/aaip-processing-information 抓取所有 AAIP 抽签事件（不做按月汇总），清洗 2025 年数据，生成事件级可视化与简易预测。

## 使用方式
- 安装依赖：`python -m pip install -r requirements.txt`
- 运行管道：`python src/aaip_pipeline.py`
- 高级事件级建模与验证：`python src/advanced_analysis.py`

输出
- 原始抓取表：`data/raw/aaip_draws_raw.csv`（官网表格原样）
- 清洗后的 2025 抽签事件：`data/processed/aaip_draws_2025.csv`（日期解析、数值化的邀请数/最低分）
- 事件级图：`reports/figures/aaip_2025_event_timeline.png`
- 事件级预测图：`reports/figures/aaip_2025_event_forecasts.png`
- 基于随机森林的事件级预测输出：`data/processed/aaip_event_forecasts_rf.csv`
- 高级事件级预测（滚动验证+多模型对比）：`data/processed/aaip_event_forecasts_advanced.csv`
- 高级建模报告（MAE/最优模型/预测表）：`reports/aaip_event_model_report.md`
- 事件间隔与累计趋势探索：`data/processed/aaip_draw_stats.csv`，`reports/figures/aaip_2025_gap_hist.png`，`reports/figures/aaip_2025_cumulative_invites.png`

## 数据质量说明
- 仅使用官方 Alberta AAIP 页面，通过 `pandas.read_html` 获取。
- 校验规则：限制年份为 2025、拒绝重复抽签日期、如果缺失邀请数量比例过高则报错。
- 页面中的 “Less than 10” 等非数字邀请数保留在 `invitation_text`，`invitations` 数值列则设为缺失以避免伪造数字。

## 预测思路（轻量版）
- 按流（stream）分组，使用事件原始日期（ordinal）做特征，线性回归预测邀请数量。
- 用历史抽签的中位间隔天数推算未来抽签日期，再生成未来若干个预测点（默认 3 个）。
- 仅在某流至少有 5 次有效抽签记录时才进行预测，避免过拟合。
- 若需更强模型，可替换为 Prophet/ARIMA 处理不规则时间序列，或用树模型（XGBoost/RandomForest）结合日期和前置窗口特征。

## 模型扩展（事件级特征和随机森林）
- 脚本：`python src/modeling.py`
- 特征：日期 ordinal、抽签间隔、邀请数的前1/2次滞后、3期滚动均值、前一期最低分。
- 评估：每个流用最后 2 次抽签作为测试集，输出线性/随机森林 MAE 对比。
- 预测：按流训练随机森林，使用中位间隔推算未来抽签日期，生成未来 3 次预测并写入 `data/processed/aaip_event_forecasts_rf.csv`。

## 高级事件级分析（多模型对比 + 滚动验证）
- 脚本：`python src/advanced_analysis.py`
- 特征：日期 ordinal、抽签间隔、邀请数滞后1/2/3、滚动均值、滞后最低分、月份、day-of-year 正余弦、事件序号。
- 评估：滚动时间序列验证（逐点前滚），对比线性回归、随机森林、梯度提升，输出 MAE。
- 预测：为每个流选取验证 MAE 最低的模型，基于历史中位间隔外推未来抽签日期，迭代滞后生成未来 3 次预测，结果写入 `data/processed/aaip_event_forecasts_advanced.csv` 并在 `reports/aaip_event_model_report.md` 记录。

## 事件级探索
- 脚本：`python src/event_insights.py`
- 内容：按流计算抽签间隔分布、总邀请/均值/中位数、缺失邀请数行数；生成间隔直方图与累计邀请趋势图。
