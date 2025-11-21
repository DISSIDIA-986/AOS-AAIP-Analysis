"""
高级事件级建模与预测
-------------------
- 仅使用官方 AAIP 处理页面抓取的抽签事件（2025）。
- 特征：日期 ordinal、抽签间隔、邀请数滞后/滚动均值、最低分滞后、季节性 day-of-year 正余弦等。
- 评估：滚动时间序列验证（逐点前滚），对比线性回归、随机森林、梯度提升。
- 预测：选取每个流的最优模型，按历史中位间隔外推未来抽签日期，迭代滞后特征生成未来 N 次预测。
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
FORECAST_OUT = Path("data/processed/aaip_event_forecasts_advanced.csv")
REPORT_OUT = Path("reports/aaip_event_model_report.md")


def load_cleaned(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load cleaned draw events."""
    df = pd.read_csv(path, parse_dates=["draw_date", "month"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create event-level features per stream."""
    parts: List[pd.DataFrame] = []
    for stream, g in df.dropna(subset=["invitations"]).groupby("stream"):
        g = g.sort_values("draw_date").copy()
        g["date_ord"] = g["draw_date"].map(pd.Timestamp.toordinal)
        gaps = g["draw_date"].diff().dt.days
        median_gap = gaps.median() if not gaps.dropna().empty else 14
        g["gap_days"] = gaps.fillna(median_gap)

        # fill scores forward/backward to stabilize lags
        g["min_score_filled"] = g["min_score"].ffill().bfill()

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

        # seasonality on day-of-year
        dayofyear = g["draw_date"].dt.dayofyear
        g["sin_doy"] = np.sin(2 * np.pi * dayofyear / 365.25)
        g["cos_doy"] = np.cos(2 * np.pi * dayofyear / 365.25)

        g["month_num"] = g["draw_date"].dt.month
        g["event_index"] = range(1, len(g) + 1)
        g["stream"] = stream
        # fill remaining NaNs in feature columns with stream medians
        feature_cols = [
            "gap_days",
            "lag1_inv",
            "lag2_inv",
            "lag3_inv",
            "roll3_inv",
            "roll5_inv",
            "lag1_score",
        ]
        medians = g[feature_cols].median(numeric_only=True)
        g[feature_cols] = g[feature_cols].fillna(medians)
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    feats = pd.concat(parts, ignore_index=True)
    # Drop rows lacking essential lags
    feats = feats.dropna(subset=["lag1_inv", "lag2_inv"])
    return feats


def rolling_backtest(
    g: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    min_train: int = 8,
) -> Tuple[float, int]:
    """
    Rolling-origin evaluation: for each point after min_train, train on history and predict next point.
    Returns (MAE, test_size). If not enough data, returns (np.nan, 0).
    """
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
    """Evaluate models per stream using rolling backtest."""
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
    ]
    models = {
        "Linear": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=400, random_state=42),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
    }
    for stream, g in feats.groupby("stream"):
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
    """Pick lowest-MAE model per stream."""
    best: Dict[str, str] = {}
    df = pd.DataFrame(metrics)
    if df.empty:
        return best
    for stream, g in df.groupby("stream"):
        best_row = g.sort_values("mae").iloc[0]
        best[stream] = str(best_row["model"])
    return best


def forecast_stream(
    stream_df: pd.DataFrame,
    best_model: str,
    feature_cols: List[str],
    draws_ahead: int = 3,
) -> List[Dict[str, object]]:
    """Train on full history then iteratively forecast future draws."""
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    model = models[best_model]
    g = stream_df.sort_values("draw_date").copy()
    if g.shape[0] < 8:
        return []

    # Fit model
    model.fit(g[feature_cols], g["invitations"])
    # Collect lag state
    lag1 = float(g.iloc[-1]["invitations"])
    lag2 = float(g.iloc[-2]["invitations"])
    lag3 = float(g.iloc[-3]["invitations"]) if g.shape[0] >= 3 else lag2
    last_score = float(g.iloc[-1]["min_score"]) if pd.notna(g.iloc[-1]["min_score"]) else np.nan
    gaps = g["draw_date"].diff().dt.days.dropna()
    median_gap = int(gaps.median()) if not gaps.empty else 14
    last_date = g["draw_date"].iat[-1]

    forecasts: List[Dict[str, object]] = []
    for step in range(1, draws_ahead + 1):
        proj_date = last_date + pd.Timedelta(days=median_gap * step)
        sin_doy = np.sin(2 * np.pi * proj_date.timetuple().tm_yday / 365.25)
        cos_doy = np.cos(2 * np.pi * proj_date.timetuple().tm_yday / 365.25)
        roll3 = np.nanmean([lag1, lag2, lag3])
        roll5 = roll3  # best approximation given short horizon
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
        }
        pred_val = float(model.predict(pd.DataFrame([feat_row]))[0])
        forecasts.append(
            {
                "stream": g["stream"].iat[0],
                "projected_date": proj_date.normalize(),
                "predicted_invitations": pred_val,
                "model": best_model,
                "median_gap_days": median_gap,
            }
        )
        # roll lags forward
        lag3, lag2, lag1 = lag2, lag1, pred_val
        last_date = proj_date
    return forecasts


def run() -> None:
    df = load_cleaned()
    feats = add_features(df)
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
    ]

    metrics = evaluate_streams(feats)
    best_map = select_best_models(metrics)

    all_forecasts: List[Dict[str, object]] = []
    for stream, model_name in best_map.items():
        g = feats.loc[feats["stream"] == stream].copy()
        fc = forecast_stream(g, model_name, feature_cols, draws_ahead=3)
        all_forecasts.extend(fc)

    # Save forecast CSV
    if all_forecasts:
        forecast_df = pd.DataFrame(all_forecasts)
        FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(FORECAST_OUT, index=False)
        print(f"Saved advanced forecasts -> {FORECAST_OUT}")
    else:
        forecast_df = pd.DataFrame()
        print("No forecasts generated (insufficient history per stream).")

    # Save report
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("# AAIP 抽签事件级建模报告（高级版）\n\n")
        f.write("数据源：阿尔伯塔省政府官网（processing page），2025 年抽签事件。\n\n")
        f.write("## 滚动验证 MAE（越低越好）\n\n")
        if metrics:
            f.write("| Stream | Model | MAE | test_size | train_events |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for m in metrics:
                f.write(
                    f"| {m['stream']} | {m['model']} | {m['mae']} | {m['test_size']} | {m['train_events']} |\n"
                )
        else:
            f.write("无可评估数据（样本不足）。\n")
        f.write("\n\n## 最优模型选择\n\n")
        if best_map:
            for stream, model in best_map.items():
                f.write(f"- {stream}: {model}\n")
        else:
            f.write("- 无可用模型（样本不足）。\n")
        f.write("\n\n## 未来抽签事件预测\n\n")
        if not forecast_df.empty:
            f.write(f"输出：`{FORECAST_OUT}`\n\n")
            f.write("| Stream | projected_date | predicted_invitations | model | median_gap_days |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for _, row in forecast_df.iterrows():
                f.write(
                    f"| {row['stream']} | {row['projected_date']} | {row['predicted_invitations']:.1f} | {row['model']} | {int(row['median_gap_days'])} |\n"
                )
        else:
            f.write("未生成预测（样本不足）。\n")
    print(f"Report saved -> {REPORT_OUT}")


if __name__ == "__main__":
    run()
