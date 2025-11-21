"""
Event-level modeling utilities for AAIP draws.

Focus: irregular draw dates, per-stream modeling using simple regressors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/aaip_draws_2025.csv")
FORECAST_OUT = Path("data/processed/aaip_event_forecasts_rf.csv")


def load_cleaned(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned 2025 draw events."""
    df = pd.read_csv(path, parse_dates=["draw_date", "month"])
    return df


def _add_lag_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add lag-based features for a single stream group."""
    g = g.sort_values("draw_date").copy()
    g["date_ord"] = g["draw_date"].map(pd.Timestamp.toordinal)
    gaps = g["date_ord"].diff()
    gap_fill = gaps.median() if not gaps.dropna().empty else 14
    g["gap_days"] = gaps.fillna(gap_fill)
    g["lag1_inv"] = g["invitations"].shift(1)
    g["lag2_inv"] = g["invitations"].shift(2)
    g["rolling_mean3"] = (
        g["invitations"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )
    g["lag1_score"] = g["min_score"].shift(1)
    return g


def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build supervised learning dataset with lag features per stream.

    Drops rows with missing invitations or missing lags to keep targets aligned.
    """
    parts: List[pd.DataFrame] = []
    for stream, g in df.dropna(subset=["invitations"]).groupby("stream"):
        g_feat = _add_lag_features(g)
        g_feat["stream"] = stream
        g_feat = g_feat.dropna(subset=["lag1_inv", "lag2_inv"])
        parts.append(g_feat)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def evaluate_models(features: pd.DataFrame) -> List[Dict[str, object]]:
    """
    Simple hold-out evaluation per stream using last 2 draws as test.

    Returns list of metrics dicts for reporting.
    """
    metrics: List[Dict[str, object]] = []
    feature_cols = ["date_ord", "gap_days", "lag1_inv", "lag2_inv", "rolling_mean3", "lag1_score"]
    for stream, g in features.groupby("stream"):
        g = g.sort_values("draw_date")
        if g.shape[0] < 6:
            continue
        train, test = g.iloc[:-2], g.iloc[-2:]
        X_train, y_train = train[feature_cols], train["invitations"]
        X_test, y_test = test[feature_cols], test["invitations"]

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)

        lin_pred = lin.predict(X_test)
        rf_pred = rf.predict(X_test)
        metrics.append(
            {
                "stream": stream,
                "test_size": len(test),
                "mae_linear": round(mean_absolute_error(y_test, lin_pred), 2),
                "mae_rf": round(mean_absolute_error(y_test, rf_pred), 2),
            }
        )
    return metrics


def forecast_next_events(df: pd.DataFrame, draws_ahead: int = 3) -> pd.DataFrame:
    """Forecast future events per stream using RandomForest and median gap for dates."""
    feature_cols = ["date_ord", "gap_days", "lag1_inv", "lag2_inv", "rolling_mean3", "lag1_score"]
    forecasts: List[Dict[str, object]] = []
    for stream, g in df.dropna(subset=["invitations"]).groupby("stream"):
        g_feat = _add_lag_features(g)
        g_feat = g_feat.dropna(subset=["lag1_inv", "lag2_inv"])
        if g_feat.shape[0] < 6:
            continue
        g_feat = g_feat.sort_values("draw_date")
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(g_feat[feature_cols], g_feat["invitations"])

        # prepare iterative lags
        last_row = g_feat.iloc[-1]
        lag1 = float(last_row["invitations"])
        lag2 = float(g_feat.iloc[-2]["invitations"])
        lag_scores = float(last_row["min_score"]) if pd.notna(last_row["min_score"]) else None
        gaps = g_feat["draw_date"].diff().dt.days.dropna()
        median_gap = int(gaps.median()) if not gaps.empty else 14
        last_date = g_feat["draw_date"].iat[-1]

        for step in range(1, draws_ahead + 1):
            proj_date = last_date + pd.Timedelta(days=median_gap * step)
            rolling_mean3 = (lag1 + lag2) / 2 if pd.notna(lag2) else lag1
            feat_row = {
                "date_ord": proj_date.toordinal(),
                "gap_days": median_gap,
                "lag1_inv": lag1,
                "lag2_inv": lag2,
                "rolling_mean3": rolling_mean3,
                "lag1_score": lag_scores,
            }
            pred_val = float(model.predict(pd.DataFrame([feat_row]))[0])
            forecasts.append(
                {
                    "stream": stream,
                    "projected_date": proj_date.normalize(),
                    "predicted_invitations": pred_val,
                    "model": "RandomForestRegressor",
                }
            )
            # roll lags forward
            lag2, lag1 = lag1, pred_val
    if not forecasts:
        return pd.DataFrame()
    return pd.DataFrame(forecasts)


def main() -> None:
    df = load_cleaned()
    features = build_supervised(df)
    metrics = evaluate_models(features)
    if metrics:
        print("Per-stream hold-out MAE (last 2 draws as test):")
        for m in metrics:
            print(m)
    else:
        print("Not enough data to evaluate models.")

    forecast_df = forecast_next_events(df)
    if not forecast_df.empty:
        FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(FORECAST_OUT, index=False)
        print(f"Saved forecasts -> {FORECAST_OUT}")
        print(forecast_df)
    else:
        print("Not enough data to forecast.")


if __name__ == "__main__":
    main()
