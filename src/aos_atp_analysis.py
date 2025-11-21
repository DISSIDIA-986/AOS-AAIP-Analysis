"""
AOS & ATP Enhanced Analysis
----------------------------
Specialized analysis and forecasting for:
- Alberta Opportunity Stream (AOS)
- Accelerated Tech Pathway (ATP)

Focuses on user-specific concerns and provides deeper insights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Import from fixed_modeling
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fixed_modeling import add_features_safe, block_bootstrap_predict

DATA_PATH = Path("data/processed/aaip_draws_2025.csv")
ATP_DATA_PATH = Path("data/processed/atp_draws_2025.csv")
OUTPUT_DIR = Path("reports/aos_atp_analysis")


def load_and_filter_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and extract AOS and ATP streams."""
    df = pd.read_csv(DATA_PATH, parse_dates=["draw_date"])

    # Filter for AOS
    aos_df = df[df["stream"] == "Alberta Opportunity Stream"].copy()

    # Load ATP separately (preserved from raw data)
    if ATP_DATA_PATH.exists():
        atp_df = pd.read_csv(ATP_DATA_PATH, parse_dates=["draw_date"])
    else:
        print("⚠️  ATP data not found, run extract_atp_data.py first")
        atp_df = pd.DataFrame()

    return aos_df, atp_df


def analyze_aos_characteristics(aos_df: pd.DataFrame) -> Dict:
    """Analyze AOS stream characteristics."""
    if aos_df.empty:
        return {"status": "No data available"}

    aos_df = aos_df.dropna(subset=["invitations"])

    analysis = {
        "total_draws": len(aos_df),
        "date_range": f"{aos_df['draw_date'].min().date()} to {aos_df['draw_date'].max().date()}",
        "invitations": {
            "mean": float(aos_df["invitations"].mean()),
            "median": float(aos_df["invitations"].median()),
            "min": int(aos_df["invitations"].min()),
            "max": int(aos_df["invitations"].max()),
            "std": float(aos_df["invitations"].std()),
            "total": int(aos_df["invitations"].sum()),
        },
        "min_score": {
            "mean": float(aos_df["min_score"].mean()) if "min_score" in aos_df.columns else None,
            "median": float(aos_df["min_score"].median()) if "min_score" in aos_df.columns else None,
            "min": int(aos_df["min_score"].min()) if "min_score" in aos_df.columns else None,
            "max": int(aos_df["min_score"].max()) if "min_score" in aos_df.columns else None,
        },
        "draw_frequency": {
            "avg_gap_days": float(aos_df["draw_date"].diff().dt.days.mean()),
            "median_gap_days": float(aos_df["draw_date"].diff().dt.days.median()),
        },
    }

    return analysis


def analyze_atp_characteristics(atp_df: pd.DataFrame) -> Dict:
    """Analyze ATP stream characteristics."""
    if atp_df.empty:
        return {"status": "No data available"}

    atp_df = atp_df.dropna(subset=["invitations"])

    analysis = {
        "total_draws": len(atp_df),
        "date_range": f"{atp_df['draw_date'].min().date()} to {atp_df['draw_date'].max().date()}",
        "invitations": {
            "mean": float(atp_df["invitations"].mean()),
            "median": float(atp_df["invitations"].median()),
            "min": int(atp_df["invitations"].min()),
            "max": int(atp_df["invitations"].max()),
            "std": float(atp_df["invitations"].std()),
            "total": int(atp_df["invitations"].sum()),
        },
        "min_score": {
            "mean": float(atp_df["min_score"].mean()) if "min_score" in atp_df.columns else None,
            "median": float(atp_df["min_score"].median()) if "min_score" in atp_df.columns else None,
            "min": int(atp_df["min_score"].min()) if "min_score" in atp_df.columns else None,
            "max": int(atp_df["min_score"].max()) if "min_score" in atp_df.columns else None,
        },
        "draw_frequency": {
            "avg_gap_days": float(atp_df["draw_date"].diff().dt.days.mean()),
            "median_gap_days": float(atp_df["draw_date"].diff().dt.days.median()),
        },
    }

    return analysis


def forecast_aos_atp(
    stream_df: pd.DataFrame,
    stream_name: str,
    draws_ahead: int = 5,
    n_bootstrap: int = 100,
) -> List[Dict]:
    """Generate forecasts for AOS or ATP with Bootstrap CI."""
    if len(stream_df) < 8:
        return []

    stream_df = stream_df.sort_values("draw_date").copy()

    # Build features
    feats = add_features_safe(stream_df)
    if feats.empty or len(feats) < 8:
        return []

    feature_cols = [
        "date_ord", "gap_days", "lag1_inv", "lag2_inv", "lag3_inv",
        "roll3_inv", "roll5_inv", "lag1_score", "sin_doy", "cos_doy",
        "month_num", "event_index", "is_holiday_week", "is_priority_sector",
        "cumulative_invitations", "gap_deviation", "is_gap_anomaly",
    ]

    # Train model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10 if stream_name == "ATP" else 15,
        min_samples_split=3,
        random_state=42 + hash(stream_name) % 100,
    )

    model.fit(feats[feature_cols], feats["invitations"])

    # Prepare forecasting
    last_date = feats["draw_date"].max()
    median_gap = feats["gap_days"].median()

    lag1 = float(feats.iloc[-1]["invitations"])
    lag2 = float(feats.iloc[-2]["invitations"]) if len(feats) >= 2 else lag1
    lag3 = float(feats.iloc[-3]["invitations"]) if len(feats) >= 3 else lag2

    last_score = feats.iloc[-1]["lag1_score"] if "lag1_score" in feats.columns else 300.0
    cumulative = feats["cumulative_invitations"].max()

    forecasts = []

    for i in range(draws_ahead):
        # Project next draw date
        proj_date = last_date + pd.Timedelta(days=median_gap * (i + 1))

        # Build feature row
        feat_row = {
            "date_ord": proj_date.toordinal(),
            "gap_days": median_gap,
            "lag1_inv": lag1,
            "lag2_inv": lag2,
            "lag3_inv": lag3,
            "roll3_inv": np.mean([lag1, lag2, lag3]),
            "roll5_inv": lag1,
            "lag1_score": last_score,
            "sin_doy": np.sin(2 * np.pi * proj_date.dayofyear / 365.25),
            "cos_doy": np.cos(2 * np.pi * proj_date.dayofyear / 365.25),
            "month_num": proj_date.month,
            "event_index": len(feats) + i,
            "is_holiday_week": 0,
            "is_priority_sector": 0,
            "cumulative_invitations": cumulative,
            "gap_deviation": 0,
            "is_gap_anomaly": 0,
        }

        # Main prediction
        pred_val = float(model.predict(pd.DataFrame([feat_row]))[0])

        # Block Bootstrap CI
        stream_random_state = 42 + hash(stream_name) % 1000
        lower_ci, median_pred, upper_ci = block_bootstrap_predict(
            model,
            feat_row,
            feats,
            feature_cols,
            n_bootstrap=n_bootstrap,
            block_size=5,
            random_state=stream_random_state + i,
        )

        forecasts.append({
            "stream": stream_name,
            "projected_date": proj_date.normalize(),
            "predicted_invitations": pred_val,
            "lower_95ci": max(0, lower_ci),
            "upper_95ci": upper_ci,
            "median_gap_days": median_gap,
        })

        # Update lags
        lag3, lag2, lag1 = lag2, lag1, pred_val
        cumulative += pred_val
        last_date = proj_date

    return forecasts


def plot_aos_atp_comparison(aos_df: pd.DataFrame, atp_df: pd.DataFrame) -> Path:
    """Plot side-by-side comparison of AOS and ATP."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AOS plot
    if not aos_df.empty:
        aos_clean = aos_df.dropna(subset=["invitations"]).sort_values("draw_date")
        axes[0].plot(aos_clean["draw_date"], aos_clean["invitations"],
                    "o-", color="#2E86AB", markersize=8, linewidth=2, label="Historical")
        axes[0].set_title("Alberta Opportunity Stream (AOS)", fontsize=14, weight="bold")
        axes[0].set_xlabel("Date", fontsize=11)
        axes[0].set_ylabel("Invitations", fontsize=11)
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ATP plot
    if not atp_df.empty:
        atp_clean = atp_df.dropna(subset=["invitations"]).sort_values("draw_date")
        axes[1].plot(atp_clean["draw_date"], atp_clean["invitations"],
                    "o-", color="#A23B72", markersize=8, linewidth=2, label="Historical")
        axes[1].set_title("Accelerated Tech Pathway (ATP)", fontsize=14, weight="bold")
        axes[1].set_xlabel("Date", fontsize=11)
        axes[1].set_ylabel("Invitations", fontsize=11)
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "aos_atp_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_aos_atp_forecasts(
    aos_df: pd.DataFrame,
    atp_df: pd.DataFrame,
    aos_forecasts: List[Dict],
    atp_forecasts: List[Dict],
) -> Path:
    """Plot forecasts with confidence intervals for AOS and ATP."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AOS plot
    if not aos_df.empty and aos_forecasts:
        aos_clean = aos_df.dropna(subset=["invitations"]).sort_values("draw_date")
        axes[0].scatter(aos_clean["draw_date"], aos_clean["invitations"],
                       label="Historical", alpha=0.7, s=80, color="#2E86AB", zorder=3)

        aos_fc = pd.DataFrame(aos_forecasts)
        axes[0].plot(aos_fc["projected_date"], aos_fc["predicted_invitations"],
                    "s--", label="Forecast", color="#F18F01", markersize=9, linewidth=2.5, zorder=4)

        axes[0].fill_between(
            aos_fc["projected_date"],
            aos_fc["lower_95ci"],
            aos_fc["upper_95ci"],
            alpha=0.25,
            color="#F18F01",
            label="95% CI",
            zorder=2,
        )

        axes[0].set_title("AOS: Historical + Forecast (with 95% CI)", fontsize=13, weight="bold")
        axes[0].set_xlabel("Date", fontsize=11)
        axes[0].set_ylabel("Invitations", fontsize=11)
        axes[0].legend(loc="best", fontsize=9)
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ATP plot
    if not atp_df.empty and atp_forecasts:
        atp_clean = atp_df.dropna(subset=["invitations"]).sort_values("draw_date")
        axes[1].scatter(atp_clean["draw_date"], atp_clean["invitations"],
                       label="Historical", alpha=0.7, s=80, color="#A23B72", zorder=3)

        atp_fc = pd.DataFrame(atp_forecasts)
        axes[1].plot(atp_fc["projected_date"], atp_fc["predicted_invitations"],
                    "s--", label="Forecast", color="#F18F01", markersize=9, linewidth=2.5, zorder=4)

        axes[1].fill_between(
            atp_fc["projected_date"],
            atp_fc["lower_95ci"],
            atp_fc["upper_95ci"],
            alpha=0.25,
            color="#F18F01",
            label="95% CI",
            zorder=2,
        )

        axes[1].set_title("ATP: Historical + Forecast (with 95% CI)", fontsize=13, weight="bold")
        axes[1].set_xlabel("Date", fontsize=11)
        axes[1].set_ylabel("Invitations", fontsize=11)
        axes[1].legend(loc="best", fontsize=9)
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "aos_atp_forecasts.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_aos_atp_score_trends(aos_df: pd.DataFrame, atp_df: pd.DataFrame) -> Path:
    """Plot minimum score trends for AOS and ATP."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AOS score trend
    if not aos_df.empty and "min_score" in aos_df.columns:
        aos_clean = aos_df.dropna(subset=["min_score"]).sort_values("draw_date")
        if not aos_clean.empty:
            axes[0].plot(aos_clean["draw_date"], aos_clean["min_score"],
                        "o-", color="#2E86AB", markersize=7, linewidth=2)
            axes[0].set_title("AOS Minimum Score Trend", fontsize=13, weight="bold")
            axes[0].set_xlabel("Date", fontsize=11)
            axes[0].set_ylabel("Minimum Score", fontsize=11)
            axes[0].grid(True, alpha=0.3)
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ATP score trend
    if not atp_df.empty and "min_score" in atp_df.columns:
        atp_clean = atp_df.dropna(subset=["min_score"]).sort_values("draw_date")
        if not atp_clean.empty:
            axes[1].plot(atp_clean["draw_date"], atp_clean["min_score"],
                        "o-", color="#A23B72", markersize=7, linewidth=2)
            axes[1].set_title("ATP Minimum Score Trend", fontsize=13, weight="bold")
            axes[1].set_xlabel("Date", fontsize=11)
            axes[1].set_ylabel("Minimum Score", fontsize=11)
            axes[1].grid(True, alpha=0.3)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "aos_atp_score_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_aos_atp_monthly_distribution(aos_df: pd.DataFrame, atp_df: pd.DataFrame) -> Path:
    """Plot monthly invitation distribution for AOS and ATP."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AOS monthly distribution
    if not aos_df.empty:
        aos_clean = aos_df.dropna(subset=["invitations"]).copy()
        aos_clean["month"] = aos_clean["draw_date"].dt.month
        monthly_aos = aos_clean.groupby("month")["invitations"].sum()

        axes[0].bar(monthly_aos.index, monthly_aos.values, color="#2E86AB", alpha=0.8)
        axes[0].set_title("AOS Monthly Invitation Distribution (2025)", fontsize=13, weight="bold")
        axes[0].set_xlabel("Month", fontsize=11)
        axes[0].set_ylabel("Total Invitations", fontsize=11)
        axes[0].set_xticks(range(1, 13))
        axes[0].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        axes[0].grid(True, alpha=0.3, axis="y")

    # ATP monthly distribution
    if not atp_df.empty:
        atp_clean = atp_df.dropna(subset=["invitations"]).copy()
        atp_clean["month"] = atp_clean["draw_date"].dt.month
        monthly_atp = atp_clean.groupby("month")["invitations"].sum()

        axes[1].bar(monthly_atp.index, monthly_atp.values, color="#A23B72", alpha=0.8)
        axes[1].set_title("ATP Monthly Invitation Distribution (2025)", fontsize=13, weight="bold")
        axes[1].set_xlabel("Month", fontsize=11)
        axes[1].set_ylabel("Total Invitations", fontsize=11)
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "aos_atp_monthly_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_report(
    aos_analysis: Dict,
    atp_analysis: Dict,
    aos_forecasts: List[Dict],
    atp_forecasts: List[Dict],
) -> Path:
    """Generate comprehensive AOS/ATP analysis report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "aos_atp_comprehensive_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AOS & ATP Comprehensive Analysis Report\n\n")
        f.write("**Generated**: 2025-11-20\n\n")
        f.write("**Focus Streams**:\n")
        f.write("- Alberta Opportunity Stream (AOS)\n")
        f.write("- Accelerated Tech Pathway (ATP)\n\n")
        f.write("---\n\n")

        # AOS Analysis
        f.write("## Alberta Opportunity Stream (AOS) Analysis\n\n")
        if "status" in aos_analysis:
            f.write(f"**Status**: {aos_analysis['status']}\n\n")
        else:
            f.write(f"### Basic Statistics\n\n")
            f.write(f"- **Total Draws**: {aos_analysis['total_draws']}\n")
            f.write(f"- **Date Range**: {aos_analysis['date_range']}\n")
            f.write(f"- **Total Invitations**: {aos_analysis['invitations']['total']:,}\n\n")

            f.write(f"### Invitation Statistics\n\n")
            f.write(f"- **Mean**: {aos_analysis['invitations']['mean']:.1f} per draw\n")
            f.write(f"- **Median**: {aos_analysis['invitations']['median']:.1f} per draw\n")
            f.write(f"- **Range**: {aos_analysis['invitations']['min']} to {aos_analysis['invitations']['max']}\n")
            f.write(f"- **Std Dev**: {aos_analysis['invitations']['std']:.1f}\n\n")

            if aos_analysis['min_score']['mean']:
                f.write(f"### Minimum Score Statistics\n\n")
                f.write(f"- **Mean**: {aos_analysis['min_score']['mean']:.1f}\n")
                f.write(f"- **Median**: {aos_analysis['min_score']['median']:.1f}\n")
                f.write(f"- **Range**: {aos_analysis['min_score']['min']} to {aos_analysis['min_score']['max']}\n\n")

            f.write(f"### Draw Frequency\n\n")
            f.write(f"- **Average Gap**: {aos_analysis['draw_frequency']['avg_gap_days']:.1f} days\n")
            f.write(f"- **Median Gap**: {aos_analysis['draw_frequency']['median_gap_days']:.1f} days\n\n")

        # AOS Forecasts
        if aos_forecasts:
            f.write("### AOS Forecasts (with 95% Confidence Intervals)\n\n")
            f.write("| Date | Predicted | 95% CI Lower | 95% CI Upper | Gap (days) |\n")
            f.write("|------|-----------|--------------|--------------|------------|\n")
            for fc in aos_forecasts:
                f.write(f"| {fc['projected_date'].date()} | {fc['predicted_invitations']:.0f} | ")
                f.write(f"{fc['lower_95ci']:.0f} | {fc['upper_95ci']:.0f} | ")
                f.write(f"{fc['median_gap_days']:.0f} |\n")
            f.write("\n")

        f.write("---\n\n")

        # ATP Analysis
        f.write("## Accelerated Tech Pathway (ATP) Analysis\n\n")
        if "status" in atp_analysis:
            f.write(f"**Status**: {atp_analysis['status']}\n\n")
        else:
            f.write(f"### Basic Statistics\n\n")
            f.write(f"- **Total Draws**: {atp_analysis['total_draws']}\n")
            f.write(f"- **Date Range**: {atp_analysis['date_range']}\n")
            f.write(f"- **Total Invitations**: {atp_analysis['invitations']['total']:,}\n\n")

            f.write(f"### Invitation Statistics\n\n")
            f.write(f"- **Mean**: {atp_analysis['invitations']['mean']:.1f} per draw\n")
            f.write(f"- **Median**: {atp_analysis['invitations']['median']:.1f} per draw\n")
            f.write(f"- **Range**: {atp_analysis['invitations']['min']} to {atp_analysis['invitations']['max']}\n")
            f.write(f"- **Std Dev**: {atp_analysis['invitations']['std']:.1f}\n\n")

            if atp_analysis['min_score']['mean']:
                f.write(f"### Minimum Score Statistics\n\n")
                f.write(f"- **Mean**: {atp_analysis['min_score']['mean']:.1f}\n")
                f.write(f"- **Median**: {atp_analysis['min_score']['median']:.1f}\n")
                f.write(f"- **Range**: {atp_analysis['min_score']['min']} to {atp_analysis['min_score']['max']}\n\n")

            f.write(f"### Draw Frequency\n\n")
            f.write(f"- **Average Gap**: {atp_analysis['draw_frequency']['avg_gap_days']:.1f} days\n")
            f.write(f"- **Median Gap**: {atp_analysis['draw_frequency']['median_gap_days']:.1f} days\n\n")

        # ATP Forecasts
        if atp_forecasts:
            f.write("### ATP Forecasts (with 95% Confidence Intervals)\n\n")
            f.write("| Date | Predicted | 95% CI Lower | 95% CI Upper | Gap (days) |\n")
            f.write("|------|-----------|--------------|--------------|------------|\n")
            for fc in atp_forecasts:
                f.write(f"| {fc['projected_date'].date()} | {fc['predicted_invitations']:.0f} | ")
                f.write(f"{fc['lower_95ci']:.0f} | {fc['upper_95ci']:.0f} | ")
                f.write(f"{fc['median_gap_days']:.0f} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## Key Insights\n\n")

        # Comparative insights
        if "status" not in aos_analysis and "status" not in atp_analysis:
            f.write("### Stream Comparison\n\n")
            f.write(f"- **AOS vs ATP Invitation Volume**: AOS averages {aos_analysis['invitations']['mean']:.1f} invitations per draw, ")
            f.write(f"while ATP averages {atp_analysis['invitations']['mean']:.1f} invitations per draw.\n")

            if aos_analysis['min_score']['mean'] and atp_analysis['min_score']['mean']:
                f.write(f"- **AOS vs ATP Score Requirements**: AOS has an average minimum score of {aos_analysis['min_score']['mean']:.1f}, ")
                f.write(f"while ATP has an average minimum score of {atp_analysis['min_score']['mean']:.1f}.\n")

            f.write(f"- **Draw Frequency**: AOS draws occur every {aos_analysis['draw_frequency']['median_gap_days']:.1f} days on average, ")
            f.write(f"while ATP draws occur every {atp_analysis['draw_frequency']['median_gap_days']:.1f} days on average.\n\n")

        f.write("### Limitations\n\n")
        f.write("- ⚠️ **Sample Size**: Limited historical data in 2025 (AOS: " + str(aos_analysis.get('total_draws', 0)) + " draws, ATP: " + str(atp_analysis.get('total_draws', 0)) + " draws)\n")
        f.write("- ⚠️ **Policy Changes**: Predictions assume policy stability; actual draws may vary\n")
        f.write("- ⚠️ **Confidence Intervals**: Wide CIs reflect uncertainty due to limited data\n\n")

        f.write("### Recommendations\n\n")
        f.write("1. **Monitor Official Announcements**: Check Alberta government website regularly\n")
        f.write("2. **Prepare Documentation**: Ensure all required documents are ready in advance\n")
        f.write("3. **Score Optimization**: Work on improving CRS/Alberta score based on historical minimums\n")
        f.write("4. **Timing Strategy**: Consider predicted draw dates for application timing\n\n")

        f.write("---\n\n")
        f.write("**Disclaimer**: These predictions are for informational purposes only and do not constitute immigration advice. ")
        f.write("Actual draw outcomes may differ based on policy changes, economic conditions, and other factors.\n")

    return report_path


def run():
    """Execute full AOS/ATP analysis workflow."""
    print("📊 Loading AOS and ATP data...")
    aos_df, atp_df = load_and_filter_data()

    print(f"✅ AOS: {len(aos_df)} draws, ATP: {len(atp_df)} draws")

    # Analysis
    print("📈 Analyzing AOS characteristics...")
    aos_analysis = analyze_aos_characteristics(aos_df)

    print("📈 Analyzing ATP characteristics...")
    atp_analysis = analyze_atp_characteristics(atp_df)

    # Forecasts
    print("🔮 Generating AOS forecasts...")
    aos_forecasts = forecast_aos_atp(aos_df, "AOS", draws_ahead=5)

    print("🔮 Generating ATP forecasts...")
    atp_forecasts = forecast_aos_atp(atp_df, "ATP", draws_ahead=5)

    # Visualizations
    print("📊 Creating comparison charts...")
    chart1 = plot_aos_atp_comparison(aos_df, atp_df)
    print(f"✅ {chart1}")

    print("📊 Creating forecast charts...")
    chart2 = plot_aos_atp_forecasts(aos_df, atp_df, aos_forecasts, atp_forecasts)
    print(f"✅ {chart2}")

    print("📊 Creating score trend charts...")
    chart3 = plot_aos_atp_score_trends(aos_df, atp_df)
    print(f"✅ {chart3}")

    print("📊 Creating monthly distribution charts...")
    chart4 = plot_aos_atp_monthly_distribution(aos_df, atp_df)
    print(f"✅ {chart4}")

    # Report
    print("📝 Generating comprehensive report...")
    report = generate_report(aos_analysis, atp_analysis, aos_forecasts, atp_forecasts)
    print(f"✅ {report}")

    print("\n✅ AOS/ATP analysis complete!")
    print(f"📂 Output directory: {OUTPUT_DIR}")

    return {
        "aos_analysis": aos_analysis,
        "atp_analysis": atp_analysis,
        "aos_forecasts": aos_forecasts,
        "atp_forecasts": atp_forecasts,
        "charts": [chart1, chart2, chart3, chart4],
        "report": report,
    }


if __name__ == "__main__":
    results = run()
