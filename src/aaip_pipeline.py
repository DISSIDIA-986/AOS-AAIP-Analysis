"""
AAIP data pipeline
------------------
Fetch AAIP processing page tables, clean draw data for 2025, validate it,
and produce basic forecasts/visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

SOURCE_URL = "https://www.alberta.ca/aaip-processing-information"
RAW_DRAW_CSV = Path("data/raw/aaip_draws_raw.csv")
PROCESSED_DRAW_CSV = Path("data/processed/aaip_draws_2025.csv")
FIGURES_DIR = Path("reports/figures")


@dataclass
class ForecastResult:
    stream: str
    predictions: List[Tuple[pd.Timestamp, float]]


def fetch_draw_table() -> pd.DataFrame:
    """Pull tables from the AAIP processing page and return the draw table."""
    tables = pd.read_html(SOURCE_URL)
    draw_df = tables[-1].copy()
    draw_df.to_csv(RAW_DRAW_CSV, index=False)
    return draw_df


def clean_draws(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, parse dates, and extract numeric fields."""
    renamed = df.rename(
        columns={
            df.columns[0]: "draw_date",
            df.columns[1]: "stream_raw",
            df.columns[2]: "min_score_raw",
            df.columns[3]: "invitation_text",
        }
    )
    renamed["draw_date"] = pd.to_datetime(renamed["draw_date"])
    renamed["year"] = renamed["draw_date"].dt.year
    renamed["month"] = renamed["draw_date"].dt.to_period("M").dt.to_timestamp()

    renamed["stream"] = renamed["stream_raw"].str.split("–").str[0].str.strip()
    renamed["stream"] = renamed["stream"].str.split("-").str[0].str.strip()

    renamed["min_score"] = (
        renamed["min_score_raw"]
        .astype(str)
        .str.extract(r"([0-9]+)")
        .astype("Int64")
    )

    invitation_text = renamed["invitation_text"].astype(str)
    less_than_mask = invitation_text.str.contains("less than", case=False, na=False)
    numeric_text = invitation_text.where(~less_than_mask)
    renamed["invitations"] = pd.to_numeric(numeric_text, errors="coerce").astype("Int64")

    cleaned = renamed.loc[renamed["year"] == 2025].copy()
    cleaned = cleaned.sort_values("draw_date")
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def validate_draws(df: pd.DataFrame) -> None:
    """Basic validation for date coverage and duplicates."""
    if df["draw_date"].dt.year.nunique() != 1 or df["draw_date"].dt.year.iloc[0] != 2025:
        raise ValueError("Dataset must only contain 2025 records.")
    if df["draw_date"].duplicated().any():
        dupes = df[df["draw_date"].duplicated()]["draw_date"].dt.date.tolist()
        raise ValueError(f"Duplicate draw dates found: {dupes}")
    if df["invitations"].isna().mean() > 0.3:
        raise ValueError("Too many missing invitation counts for reliable analysis.")


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate invitation counts monthly by stream (optional helper)."""
    monthly = (
        df.dropna(subset=["invitations"])
        .groupby(["month", "stream"], as_index=False)["invitations"]
        .sum()
    )
    monthly = monthly.sort_values(["month", "stream"])
    return monthly


def forecast_stream(monthly: pd.DataFrame, months_ahead: int = 3) -> List[ForecastResult]:
    """Fit simple linear models per stream and forecast future months."""
    results: List[ForecastResult] = []
    for stream, group in monthly.groupby("stream"):
        if group.shape[0] < 3:
            continue
        group = group.sort_values("month")
        group["month_num"] = range(1, len(group) + 1)
        model = LinearRegression()
        model.fit(group[["month_num"]], group["invitations"])

        last_month_num = group["month_num"].iat[-1]
        last_month = group["month"].iat[-1]
        preds: List[Tuple[pd.Timestamp, float]] = []
        for step in range(1, months_ahead + 1):
            month_num = last_month_num + step
            future_month = (last_month + pd.offsets.MonthBegin(step)).to_period("M").to_timestamp()
            pred_input = pd.DataFrame({"month_num": [month_num]})
            pred_val = float(model.predict(pred_input)[0])
            preds.append((future_month, pred_val))
        results.append(ForecastResult(stream=stream, predictions=preds))
    return results


def forecast_stream_events(df: pd.DataFrame, draws_ahead: int = 3) -> List[ForecastResult]:
    """
    Per-stream forecasting on irregular draw dates.

    Uses median day gap between past draws to project future draw dates,
    and a simple linear regression on date ordinal to forecast invitations.
    Only runs when a stream has at least five draws to avoid overfitting noise.
    """
    results: List[ForecastResult] = []
    for stream, group in df.groupby("stream"):
        group = group.dropna(subset=["invitations"]).sort_values("draw_date")
        if group.shape[0] < 5:
            continue

        ordinals = group["draw_date"].map(pd.Timestamp.toordinal)
        model = LinearRegression()
        model.fit(ordinals.to_frame(name="date_ord"), group["invitations"])

        # median gap to project future draw dates
        gaps = ordinals.diff().dropna()
        median_gap = int(gaps.median()) if not gaps.empty else 14
        last_date = group["draw_date"].iat[-1]
        preds: List[Tuple[pd.Timestamp, float]] = []
        for step in range(1, draws_ahead + 1):
            future_date = last_date + pd.Timedelta(days=median_gap * step)
            pred_val = float(model.predict([[future_date.toordinal()]])[0])
            preds.append((future_date, pred_val))
        results.append(ForecastResult(stream=stream, predictions=preds))
    return results


def plot_event_timeline(df: pd.DataFrame) -> Path:
    """Plot actual draw events on their exact dates."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df,
        x="draw_date",
        y="invitations",
        hue="stream",
        alpha=0.8,
    )
    plt.title("AAIP Invitations per Draw (2025)")
    plt.xlabel("Draw date")
    plt.ylabel("Invitations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path = FIGURES_DIR / "aaip_2025_event_timeline.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_stream_event_forecasts(df: pd.DataFrame, forecasts: List[ForecastResult]) -> Path:
    """Plot event-level history with projected future draw points."""
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "stream": row["stream"],
                "date": row["draw_date"],
                "invitations": row["invitations"],
                "is_forecast": False,
            }
        )
    for forecast in forecasts:
        for date, value in forecast.predictions:
            records.append(
                {
                    "stream": forecast.stream,
                    "date": date,
                    "invitations": value,
                    "is_forecast": True,
                }
            )
    plot_df = pd.DataFrame.from_records(records)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x="date",
        y="invitations",
        hue="stream",
        style="is_forecast",
        markers=True,
        s=70,
        alpha=0.85,
    )
    plt.title("AAIP Draws and Forecasted Future Events (2025)")
    plt.xlabel("Date")
    plt.ylabel("Invitations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path = FIGURES_DIR / "aaip_2025_event_forecasts.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> Dict[str, Path]:
    RAW_DRAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DRAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    draw_df = fetch_draw_table()
    cleaned = clean_draws(draw_df)
    validate_draws(cleaned)

    cleaned.to_csv(PROCESSED_DRAW_CSV, index=False)
    forecasts = forecast_stream_events(cleaned)

    event_plot = plot_event_timeline(cleaned)
    forecast_plot = plot_stream_event_forecasts(cleaned, forecasts)
    return {
        "raw_draws": RAW_DRAW_CSV,
        "cleaned_draws": PROCESSED_DRAW_CSV,
        "event_plot": event_plot,
        "forecast_plot": forecast_plot,
    }


if __name__ == "__main__":
    outputs = main()
    for name, path in outputs.items():
        print(f"{name}: {path}")
