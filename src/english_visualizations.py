"""
English Visualizations - Fixed Charts
--------------------------------------
All charts in English to avoid font issues.
Uses the enhanced forecasts with prediction intervals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path("data/processed/aaip_draws_2025.csv")
FORECAST_PATH = Path("data/processed/aaip_forecasts_fixed.csv")  # Use FIXED forecasts
FIGURES_DIR = Path("reports/figures/english")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical data and forecasts."""
    df = pd.read_csv(DATA_PATH, parse_dates=["draw_date"])
    try:
        forecast_df = pd.read_csv(FORECAST_PATH, parse_dates=["projected_date"])
    except FileNotFoundError:
        forecast_df = pd.DataFrame()
    return df, forecast_df


def plot_forecast_with_intervals(df: pd.DataFrame, forecast_df: pd.DataFrame) -> Path:
    """Plot forecasts with 95% confidence intervals."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    streams = forecast_df["stream"].unique() if not forecast_df.empty else []

    for idx, stream in enumerate(streams[:4]):
        ax = axes[idx]

        # Historical data
        hist = df[df["stream"] == stream].dropna(subset=["invitations"]).sort_values("draw_date")
        ax.scatter(hist["draw_date"], hist["invitations"],
                   label="Historical Data", alpha=0.7, s=60, color="#2E86AB")

        # Forecasts
        fc = forecast_df[forecast_df["stream"] == stream].sort_values("projected_date")
        if not fc.empty:
            ax.plot(fc["projected_date"], fc["predicted_invitations"],
                    "o-", label="Predicted", color="#A23B72", markersize=8, linewidth=2)

            # 95% CI bands
            ax.fill_between(
                fc["projected_date"],
                fc["lower_95ci"],
                fc["upper_95ci"],
                alpha=0.3,
                color="#A23B72",
                label="95% Confidence Interval"
            )

            # Holiday markers
            holiday_fc = fc[fc["is_holiday_week"] == 1]
            if not holiday_fc.empty:
                ax.scatter(holiday_fc["projected_date"], holiday_fc["predicted_invitations"],
                          marker="*", s=300, color="red", zorder=5, label="Holiday Week")

        ax.set_title(f"{stream}", fontsize=12, weight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Invitations", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(streams), 4):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = FIGURES_DIR / "forecast_with_intervals.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_feature_importance(df: pd.DataFrame) -> Path:
    """Feature importance analysis with Random Forest."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.fixed_modeling import add_features_safe

    feats = add_features_safe(df)
    if feats.empty:
        return FIGURES_DIR / "feature_importance.png"

    feature_cols = [
        "date_ord", "gap_days", "lag1_inv", "lag2_inv", "lag3_inv",
        "roll3_inv", "roll5_inv", "lag1_score", "sin_doy", "cos_doy",
        "month_num", "event_index", "is_holiday_week", "is_priority_sector",
        "cumulative_invitations", "gap_deviation", "is_gap_anomaly"
    ]

    importance_data = []

    for stream, g in feats.groupby("stream"):
        if len(g) < 15:
            continue

        g = g.dropna(subset=feature_cols + ["invitations"])
        if g.empty:
            continue

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(g[feature_cols], g["invitations"])

        for feat, imp in zip(feature_cols, model.feature_importances_):
            importance_data.append({"stream": stream, "feature": feat, "importance": imp})

    if not importance_data:
        return FIGURES_DIR / "feature_importance.png"

    imp_df = pd.DataFrame(importance_data)

    # Average importance
    avg_imp = imp_df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(12)

    # Rename features to readable English
    feature_names = {
        "lag1_inv": "Lag-1 Invitations",
        "roll3_inv": "3-Period Rolling Mean",
        "date_ord": "Date Ordinal",
        "cumulative_invitations": "Cumulative Invitations",
        "lag2_inv": "Lag-2 Invitations",
        "gap_days": "Gap Between Draws (days)",
        "roll5_inv": "5-Period Rolling Mean",
        "lag1_score": "Lag-1 Min Score",
        "event_index": "Event Sequence Index",
        "month_num": "Month Number",
        "is_holiday_week": "Holiday Week Indicator",
        "is_priority_sector": "Priority Sector Indicator",
        "gap_deviation": "Gap Deviation from Median",
        "sin_doy": "Day of Year (Sine)",
        "cos_doy": "Day of Year (Cosine)",
        "lag3_inv": "Lag-3 Invitations",
        "is_gap_anomaly": "Gap Anomaly Indicator",
    }

    avg_imp.index = [feature_names.get(f, f) for f in avg_imp.index]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_imp.values, y=avg_imp.index, palette="viridis")
    plt.title("Feature Importance Analysis (Random Forest Average)", fontsize=14, weight="bold")
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    output_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_model_performance_comparison() -> Path:
    """Model performance comparison chart."""
    try:
        # Use FIXED model report
        with open("reports/fixed_model_report.md", "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        metrics = []
        in_table = False
        for line in lines:
            if "| Stream | Model | MAE" in line:
                in_table = True
                continue
            if in_table and line.startswith("|") and "---" not in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 5:
                    try:
                        metrics.append({
                            "stream": parts[0],
                            "model": parts[1],
                            "mae": float(parts[2])
                        })
                    except ValueError:
                        continue
            elif in_table and not line.startswith("|"):
                break

        if not metrics:
            return FIGURES_DIR / "model_comparison.png"

        df = pd.DataFrame(metrics)

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))

        streams = df["stream"].unique()
        x = np.arange(len(streams))
        width = 0.25

        models = df["model"].unique()
        colors = ["#2E86AB", "#A23B72", "#F18F01"]

        for i, model in enumerate(models):
            model_data = df[df["model"] == model].sort_values("stream")
            mae_values = [model_data[model_data["stream"] == s]["mae"].values[0]
                         if s in model_data["stream"].values else 0
                         for s in streams]
            ax.bar(x + i * width, mae_values, width, label=model, color=colors[i % len(colors)])

        ax.set_xlabel("Stream Type", fontsize=12, weight="bold")
        ax.set_ylabel("MAE (Lower is Better)", fontsize=12, weight="bold")
        ax.set_title("Model Performance Comparison (Rolling Time Series Validation)", fontsize=14, weight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(streams, rotation=15, ha="right")
        ax.legend(title="Model", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = FIGURES_DIR / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path

    except Exception as e:
        print(f"⚠️  Model comparison chart generation failed: {e}")
        return FIGURES_DIR / "model_comparison.png"


def plot_stream_trends_combined(df: pd.DataFrame, forecast_df: pd.DataFrame) -> Path:
    """Combined stream trends with forecasts and CIs."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))

    main_streams = [
        "Alberta Express Entry Stream",
        "Alberta Opportunity Stream",
        "Dedicated Health Care Pathway"
    ]

    colors = {
        "Alberta Express Entry Stream": "#2E86AB",
        "Alberta Opportunity Stream": "#A23B72",
        "Dedicated Health Care Pathway": "#F18F01"
    }

    for stream in main_streams:
        # Historical
        hist = df[df["stream"] == stream].dropna(subset=["invitations"]).sort_values("draw_date")
        if not hist.empty:
            ax.plot(hist["draw_date"], hist["invitations"],
                   "o-", label=f"{stream} (Historical)",
                   color=colors.get(stream, "#666"), alpha=0.7, linewidth=2, markersize=5)

        # Forecasts
        if not forecast_df.empty:
            fc = forecast_df[forecast_df["stream"] == stream].sort_values("projected_date")
            if not fc.empty:
                ax.plot(fc["projected_date"], fc["predicted_invitations"],
                       "s--", label=f"{stream} (Forecast)",
                       color=colors.get(stream, "#666"), linewidth=2, markersize=7)

                # CI bands
                ax.fill_between(
                    fc["projected_date"],
                    fc["lower_95ci"],
                    fc["upper_95ci"],
                    alpha=0.15,
                    color=colors.get(stream, "#666")
                )

    ax.set_title("AAIP Main Streams: Draw Trends & Forecasts (2025)", fontsize=16, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Invitations", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / "stream_trends_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_residual_diagnostics(df: pd.DataFrame) -> Path:
    """Residual diagnostic plots."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.fixed_modeling import add_features_safe

    feats = add_features_safe(df)
    if feats.empty:
        return FIGURES_DIR / "residual_diagnostics.png"

    feature_cols = [
        "date_ord", "gap_days", "lag1_inv", "lag2_inv", "lag3_inv",
        "roll3_inv", "roll5_inv", "lag1_score", "sin_doy", "cos_doy",
        "month_num", "event_index", "is_holiday_week", "is_priority_sector",
        "cumulative_invitations", "gap_deviation", "is_gap_anomaly"
    ]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    residuals_all = []

    for stream, g in feats.groupby("stream"):
        if len(g) < 15:
            continue

        g = g.dropna(subset=feature_cols + ["invitations"]).sort_values("draw_date")
        if g.empty:
            continue

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(g[feature_cols], g["invitations"])
        predictions = model.predict(g[feature_cols])
        residuals = g["invitations"].values - predictions

        residuals_all.extend(residuals)

        # Plot 1: Residuals vs Predicted
        axes[0, 0].scatter(predictions, residuals, alpha=0.6, s=50, label=stream)

    if residuals_all:
        # Plot 1 settings
        axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel("Predicted Values", fontsize=11)
        axes[0, 0].set_ylabel("Residuals", fontsize=11)
        axes[0, 0].set_title("Residuals vs Predicted Values", fontsize=12, weight="bold")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: Residual histogram
        axes[0, 1].hist(residuals_all, bins=30, color="#2E86AB", alpha=0.7, edgecolor="black")
        axes[0, 1].axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("Residuals", fontsize=11)
        axes[0, 1].set_ylabel("Frequency", fontsize=11)
        axes[0, 1].set_title("Residual Distribution", fontsize=12, weight="bold")
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Q-Q plot
        from scipy import stats
        stats.probplot(residuals_all, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normality Test)", fontsize=12, weight="bold")
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: Residual time series
        axes[1, 1].plot(residuals_all, "o-", alpha=0.6, color="#A23B72", markersize=4)
        axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[1, 1].set_xlabel("Observation Index", fontsize=11)
        axes[1, 1].set_ylabel("Residuals", fontsize=11)
        axes[1, 1].set_title("Residual Time Series", fontsize=12, weight="bold")
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "residual_diagnostics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def run() -> Dict[str, Path]:
    """Generate all English visualizations."""
    df, forecast_df = load_data()

    outputs = {}

    print("📊 Generating forecast intervals chart...")
    outputs["forecast_intervals"] = plot_forecast_with_intervals(df, forecast_df)

    print("📊 Generating feature importance chart...")
    outputs["feature_importance"] = plot_feature_importance(df)

    print("📊 Generating model comparison chart...")
    outputs["model_comparison"] = plot_model_performance_comparison()

    print("📊 Generating combined trends chart...")
    outputs["stream_trends"] = plot_stream_trends_combined(df, forecast_df)

    print("📊 Generating residual diagnostics chart...")
    outputs["residual_diagnostics"] = plot_residual_diagnostics(df)

    return outputs


if __name__ == "__main__":
    results = run()
    print("\n✅ All English charts generated:")
    for name, path in results.items():
        print(f"  - {name}: {path}")
