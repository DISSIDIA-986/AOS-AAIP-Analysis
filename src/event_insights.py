"""
事件级探索性分析：
- 计算按流的抽签间隔分布/摘要
- 绘制间隔直方图与累计邀请人数曲线
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/processed/aaip_draws_2025.csv")
STATS_OUT = Path("data/processed/aaip_draw_stats.csv")
FIG_DIR = Path("reports/figures")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["draw_date", "month"])


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap stats and invitation summaries per stream."""
    rows = []
    for stream, g in df.dropna(subset=["draw_date"]).groupby("stream"):
        g = g.sort_values("draw_date")
        gaps = g["draw_date"].diff().dt.days.dropna()
        rows.append(
            {
                "stream": stream,
                "events": len(g),
                "median_gap_days": gaps.median() if not gaps.empty else None,
                "min_gap_days": gaps.min() if not gaps.empty else None,
                "max_gap_days": gaps.max() if not gaps.empty else None,
                "total_invitations": g["invitations"].sum(skipna=True),
                "missing_invitation_rows": g["invitations"].isna().sum(),
                "mean_invitation": g["invitations"].mean(skipna=True),
                "median_invitation": g["invitations"].median(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def plot_gap_histogram(df: pd.DataFrame) -> Path:
    """Plot inter-draw day gap distribution."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    gap_records = []
    for stream, g in df.groupby("stream"):
        gaps = g.sort_values("draw_date")["draw_date"].diff().dt.days.dropna()
        if gaps.empty:
            continue
        temp = pd.DataFrame({"gap_days": gaps, "stream": stream})
        gap_records.append(temp)
    if not gap_records:
        return FIG_DIR / "aaip_2025_gap_hist.png"
    gap_df = pd.concat(gap_records, ignore_index=True)
    sns.histplot(data=gap_df, x="gap_days", hue="stream", bins=20, multiple="stack")
    plt.title("AAIP draw gap distribution (days)")
    plt.xlabel("Gap between draws (days)")
    plt.ylabel("Count")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "aaip_2025_gap_hist.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_cumulative_invites(df: pd.DataFrame) -> Path:
    """Plot cumulative invites over time per stream."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    records = []
    for stream, g in df.dropna(subset=["invitations"]).groupby("stream"):
        g = g.sort_values("draw_date").copy()
        g["cumulative_invites"] = g["invitations"].cumsum()
        g["stream"] = stream
        records.append(g)
    if not records:
        return FIG_DIR / "aaip_2025_cumulative_invites.png"
    plot_df = pd.concat(records, ignore_index=True)
    sns.lineplot(
        data=plot_df,
        x="draw_date",
        y="cumulative_invites",
        hue="stream",
        marker="o",
    )
    plt.title("AAIP cumulative invitations (event-level)")
    plt.xlabel("Draw date")
    plt.ylabel("Cumulative invitations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "aaip_2025_cumulative_invites.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def run() -> Dict[str, Path]:
    df = load_data()
    stats = compute_stats(df)
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(STATS_OUT, index=False)
    gap_plot = plot_gap_histogram(df)
    cum_plot = plot_cumulative_invites(df)
    return {"stats": STATS_OUT, "gap_plot": gap_plot, "cumulative_plot": cum_plot}


if __name__ == "__main__":
    outputs = run()
    for k, v in outputs.items():
        print(f"{k}: {v}")
