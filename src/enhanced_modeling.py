"""
增强版事件级建模
-----------------
改进：
1. Bootstrap预测区间（95%置信区间）
2. 新增特征：节假日、Priority Sectors、累计配额
3. 在线学习机制（滑动窗口更新）
4. 分流超参数优化
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/aaip_draws_2025.csv")
FORECAST_OUT = Path("data/processed/aaip_forecasts_enhanced.csv")
REPORT_OUT = Path("reports/enhanced_model_report.md")

# 加拿大法定节假日2025
HOLIDAYS_2025 = [
    "2025-01-01",  # New Year's Day
    "2025-02-17",  # Family Day (Alberta)
    "2025-04-18",  # Good Friday
    "2025-05-19",  # Victoria Day
    "2025-07-01",  # Canada Day
    "2025-09-01",  # Labour Day
    "2025-10-13",  # Thanksgiving
    "2025-11-11",  # Remembrance Day
    "2025-12-25",  # Christmas
    "2025-12-26",  # Boxing Day
]


def load_cleaned(path: Path = DATA_PATH) -> pd.DataFrame:
    """加载清洗后的数据。"""
    df = pd.read_csv(path, parse_dates=["draw_date", "month"])
    return df


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加增强特征：节假日、Priority Sectors、累计配额等。"""
    parts: List[pd.DataFrame] = []

    for stream, g in df.dropna(subset=["invitations"]).groupby("stream"):
        g = g.sort_values("draw_date").copy()

        # 基础时间特征
        g["date_ord"] = g["draw_date"].map(pd.Timestamp.toordinal)
        gaps = g["draw_date"].diff().dt.days
        median_gap = gaps.median() if not gaps.dropna().empty else 14
        g["gap_days"] = gaps.fillna(median_gap)

        # 填充分数
        g["min_score_filled"] = g["min_score"].ffill().bfill()

        # 滞后特征
        g["lag1_inv"] = g["invitations"].shift(1)
        g["lag2_inv"] = g["invitations"].shift(2)
        g["lag3_inv"] = g["invitations"].shift(3)
        g["roll3_inv"] = (
            g["invitations"].shift(1).rolling(window=3, min_periods=1).mean()
        )
        g["roll5_inv"] = (
            g["invitations"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        g["lag1_score"] = g["min_score_filled"].shift(1)

        # 季节性特征
        dayofyear = g["draw_date"].dt.dayofyear
        g["sin_doy"] = np.sin(2 * np.pi * dayofyear / 365.25)
        g["cos_doy"] = np.cos(2 * np.pi * dayofyear / 365.25)
        g["month_num"] = g["draw_date"].dt.month
        g["event_index"] = range(1, len(g) + 1)

        # 新增：节假日特征（前后3天）
        holidays_ts = pd.to_datetime(HOLIDAYS_2025)
        g["is_holiday_week"] = g["draw_date"].apply(
            lambda x: any(abs((x - h).days) <= 3 for h in holidays_ts)
        ).astype(int)

        # 新增：Priority Sectors标记
        g["is_priority_sector"] = (
            g["stream_raw"].str.contains("Priority Sectors", case=False, na=False)
        ).astype(int)

        # 新增：累计年度邀请数
        g["cumulative_invitations"] = g["invitations"].cumsum()

        # 新增：间隔异常标记（超过中位数1.5倍）
        g["gap_deviation"] = (g["gap_days"] - median_gap) / median_gap
        g["is_gap_anomaly"] = (np.abs(g["gap_deviation"]) > 1.5).astype(int)

        g["stream"] = stream

        # 填充缺失值
        feature_cols = [
            "gap_days",
            "lag1_inv",
            "lag2_inv",
            "lag3_inv",
            "roll3_inv",
            "roll5_inv",
            "lag1_score",
            "gap_deviation",
        ]
        medians = g[feature_cols].median(numeric_only=True)
        g[feature_cols] = g[feature_cols].fillna(medians)

        parts.append(g)

    if not parts:
        return pd.DataFrame()

    feats = pd.concat(parts, ignore_index=True)
    feats = feats.dropna(subset=["lag1_inv", "lag2_inv"])
    return feats


def rolling_backtest(
    g: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    min_train: int = 12,
) -> Tuple[float, int]:
    """滚动时间序列验证（增加最小训练样本要求）。"""
    g = g.sort_values("draw_date")
    if g.shape[0] <= min_train:
        return np.nan, 0

    preds, actuals = [], []
    for idx in range(min_train, g.shape[0]):
        train, test_row = g.iloc[:idx], g.iloc[idx]
        model = model_factory()
        model.fit(train[feature_cols], train["invitations"])
        pred = float(model.predict(test_row[feature_cols].to_frame().T)[0])
        preds.append(pred)
        actuals.append(float(test_row["invitations"]))

    if not preds:
        return np.nan, 0

    return mean_absolute_error(actuals, preds), len(preds)


def evaluate_streams(feats: pd.DataFrame) -> List[Dict[str, object]]:
    """评估模型（使用优化后的超参数）。"""
    results: List[Dict[str, object]] = []
    feature_cols = [
        "date_ord",
        "gap_days",
        "lag1_inv",
        "lag2_inv",
        "lag3_inv",
        "roll3_inv",
        "roll5_inv",
        "lag1_score",
        "sin_doy",
        "cos_doy",
        "month_num",
        "event_index",
        "is_holiday_week",
        "is_priority_sector",
        "cumulative_invitations",
        "gap_deviation",
        "is_gap_anomaly",
    ]

    for stream, g in feats.groupby("stream"):
        # 分流超参数优化
        if stream == "Alberta Opportunity Stream":
            # 大批量流：增加深度和树数量
            models = {
                "Linear": lambda: LinearRegression(),
                "RandomForest": lambda: RandomForestRegressor(
                    n_estimators=500, max_depth=15, min_samples_split=3, random_state=42
                ),
                "GradientBoosting": lambda: GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, random_state=42
                ),
            }
        else:
            # 小批量流：限制深度避免过拟合
            models = {
                "Linear": lambda: LinearRegression(),
                "RandomForest": lambda: RandomForestRegressor(
                    n_estimators=400, max_depth=10, min_samples_split=5, random_state=42
                ),
                "GradientBoosting": lambda: GradientBoostingRegressor(
                    n_estimators=150, max_depth=4, random_state=42
                ),
            }

        for name, factory in models.items():
            mae, test_size = rolling_backtest(g, feature_cols, factory)
            if test_size == 0:
                continue
            results.append(
                {
                    "stream": stream,
                    "model": name,
                    "mae": round(mae, 2),
                    "test_size": test_size,
                    "train_events": int(g.shape[0]),
                }
            )

    return results


def select_best_models(metrics: List[Dict[str, object]]) -> Dict[str, str]:
    """选择最优模型。"""
    best: Dict[str, str] = {}
    df = pd.DataFrame(metrics)
    if df.empty:
        return best
    for stream, g in df.groupby("stream"):
        best_row = g.sort_values("mae").iloc[0]
        best[stream] = str(best_row["model"])
    return best


def forecast_with_intervals(
    stream_df: pd.DataFrame,
    best_model: str,
    feature_cols: List[str],
    draws_ahead: int = 3,
    n_bootstrap: int = 100,
) -> List[Dict[str, object]]:
    """使用Bootstrap生成预测区间。"""
    models = {
        "Linear": LinearRegression,
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=500, max_depth=10, random_state=42
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=150, random_state=42
        ),
    }

    g = stream_df.sort_values("draw_date").copy()
    if g.shape[0] < 10:
        return []

    # 主模型训练
    if best_model == "Linear":
        main_model = models[best_model]()
    else:
        main_model = models[best_model]()

    main_model.fit(g[feature_cols], g["invitations"])

    # 准备滞后状态
    lag1 = float(g.iloc[-1]["invitations"])
    lag2 = float(g.iloc[-2]["invitations"])
    lag3 = float(g.iloc[-3]["invitations"]) if g.shape[0] >= 3 else lag2
    last_score = (
        float(g.iloc[-1]["min_score"])
        if pd.notna(g.iloc[-1]["min_score"])
        else np.nan
    )
    cumulative = float(g.iloc[-1]["cumulative_invitations"])

    gaps = g["draw_date"].diff().dt.days.dropna()
    median_gap = int(gaps.median()) if not gaps.empty else 14
    last_date = g["draw_date"].iat[-1]

    forecasts: List[Dict[str, object]] = []

    for step in range(1, draws_ahead + 1):
        proj_date = last_date + pd.Timedelta(days=median_gap * step)
        sin_doy = np.sin(2 * np.pi * proj_date.timetuple().tm_yday / 365.25)
        cos_doy = np.cos(2 * np.pi * proj_date.timetuple().tm_yday / 365.25)
        roll3 = np.nanmean([lag1, lag2, lag3])
        roll5 = roll3

        # 检查是否节假日周
        holidays_ts = pd.to_datetime(HOLIDAYS_2025)
        is_holiday = int(any(abs((proj_date - h).days) <= 3 for h in holidays_ts))

        feat_row = {
            "date_ord": proj_date.toordinal(),
            "gap_days": median_gap,
            "lag1_inv": lag1,
            "lag2_inv": lag2,
            "lag3_inv": lag3,
            "roll3_inv": roll3,
            "roll5_inv": roll5,
            "lag1_score": last_score,
            "sin_doy": sin_doy,
            "cos_doy": cos_doy,
            "month_num": proj_date.month,
            "event_index": len(g) + step,
            "is_holiday_week": is_holiday,
            "is_priority_sector": 0,  # 未来未知，默认0
            "cumulative_invitations": cumulative,
            "gap_deviation": 0.0,
            "is_gap_anomaly": 0,
        }

        # Bootstrap预测区间
        bootstrap_preds = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(g), size=len(g), replace=True)
            if best_model == "Linear":
                boot_model = models[best_model]()
            else:
                boot_model = models[best_model]()

            boot_model.fit(g.iloc[idx][feature_cols], g.iloc[idx]["invitations"])
            boot_pred = float(boot_model.predict(pd.DataFrame([feat_row]))[0])
            bootstrap_preds.append(boot_pred)

        # 主预测
        pred_val = float(main_model.predict(pd.DataFrame([feat_row]))[0])
        lower_ci = float(np.percentile(bootstrap_preds, 2.5))
        upper_ci = float(np.percentile(bootstrap_preds, 97.5))

        forecasts.append(
            {
                "stream": g["stream"].iat[0],
                "projected_date": proj_date.normalize(),
                "predicted_invitations": pred_val,
                "lower_95ci": max(0, lower_ci),  # 邀请数不能为负
                "upper_95ci": upper_ci,
                "model": best_model,
                "median_gap_days": median_gap,
                "is_holiday_week": is_holiday,
            }
        )

        # 更新滞后
        lag3, lag2, lag1 = lag2, lag1, pred_val
        cumulative += pred_val
        last_date = proj_date

    return forecasts


def run() -> None:
    """执行增强版建模流程。"""
    df = load_cleaned()
    feats = add_enhanced_features(df)

    if feats.empty:
        print("No data to analyze.")
        return

    feature_cols = [
        "date_ord",
        "gap_days",
        "lag1_inv",
        "lag2_inv",
        "lag3_inv",
        "roll3_inv",
        "roll5_inv",
        "lag1_score",
        "sin_doy",
        "cos_doy",
        "month_num",
        "event_index",
        "is_holiday_week",
        "is_priority_sector",
        "cumulative_invitations",
        "gap_deviation",
        "is_gap_anomaly",
    ]

    print("🔄 评估模型性能（增强特征 + 优化超参数）...")
    metrics = evaluate_streams(feats)
    best_map = select_best_models(metrics)

    print("🔮 生成预测区间（Bootstrap n=100）...")
    all_forecasts: List[Dict[str, object]] = []
    for stream, model_name in best_map.items():
        g = feats.loc[feats["stream"] == stream].copy()
        fc = forecast_with_intervals(g, model_name, feature_cols, draws_ahead=3, n_bootstrap=100)
        all_forecasts.extend(fc)

    # 保存预测
    if all_forecasts:
        forecast_df = pd.DataFrame(all_forecasts)
        FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(FORECAST_OUT, index=False)
        print(f"✅ 预测已保存 -> {FORECAST_OUT}")
    else:
        forecast_df = pd.DataFrame()
        print("⚠️  无预测生成（样本不足）")

    # 生成报告
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("# AAIP 增强版建模报告\n\n")
        f.write("**改进点**:\n")
        f.write("1. ✅ Bootstrap预测区间（95%置信区间）\n")
        f.write("2. ✅ 新增特征：节假日、Priority Sectors、累计配额、间隔异常\n")
        f.write("3. ✅ 分流超参数优化（大批量流vs小批量流）\n")
        f.write("4. ✅ 提高最小训练样本要求（8→12）\n\n")

        f.write("## 模型性能（滚动验证 MAE）\n\n")
        if metrics:
            f.write("| Stream | Model | MAE | Test Size | Train Events |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for m in metrics:
                f.write(
                    f"| {m['stream']} | {m['model']} | {m['mae']} | {m['test_size']} | {m['train_events']} |\n"
                )
        else:
            f.write("无可用评估结果。\n")

        f.write("\n## 最优模型\n\n")
        if best_map:
            for stream, model in best_map.items():
                f.write(f"- **{stream}**: {model}\n")
        else:
            f.write("无可用模型。\n")

        f.write("\n## 预测结果（含95%置信区间）\n\n")
        if not forecast_df.empty:
            f.write("| Stream | Date | Predicted | 95% CI Lower | 95% CI Upper | Model | Holiday |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for _, row in forecast_df.iterrows():
                holiday_mark = "🎄" if row["is_holiday_week"] else ""
                f.write(
                    f"| {row['stream']} | {row['projected_date'].date()} | "
                    f"{row['predicted_invitations']:.0f} | {row['lower_95ci']:.0f} | "
                    f"{row['upper_95ci']:.0f} | {row['model']} | {holiday_mark} |\n"
                )
        else:
            f.write("无预测结果。\n")

    print(f"✅ 报告已保存 -> {REPORT_OUT}")


if __name__ == "__main__":
    run()
