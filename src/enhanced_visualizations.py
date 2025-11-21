"""
增强版可视化
-----------
生成改进的图表：
1. 预测区间带状图（置信区间阴影）
2. 特征重要性分析图
3. 模型性能对比图
4. 分流预测趋势图（含节假日标记）
5. 残差诊断图
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path("data/processed/aaip_draws_2025.csv")
FORECAST_PATH = Path("data/processed/aaip_forecasts_enhanced.csv")
FIGURES_DIR = Path("reports/figures/enhanced")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载历史数据和预测结果。"""
    df = pd.read_csv(DATA_PATH, parse_dates=["draw_date"])
    try:
        forecast_df = pd.read_csv(FORECAST_PATH, parse_dates=["projected_date"])
    except FileNotFoundError:
        forecast_df = pd.DataFrame()
    return df, forecast_df


def plot_forecast_with_intervals(df: pd.DataFrame, forecast_df: pd.DataFrame) -> Path:
    """绘制预测区间带状图。"""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    streams = forecast_df["stream"].unique() if not forecast_df.empty else []

    for idx, stream in enumerate(streams[:4]):  # 最多4个流
        ax = axes[idx]

        # 历史数据
        hist = df[df["stream"] == stream].dropna(subset=["invitations"]).sort_values("draw_date")
        ax.scatter(hist["draw_date"], hist["invitations"],
                   label="历史数据", alpha=0.7, s=60, color="#2E86AB")

        # 预测数据
        fc = forecast_df[forecast_df["stream"] == stream].sort_values("projected_date")
        if not fc.empty:
            ax.plot(fc["projected_date"], fc["predicted_invitations"],
                    "o-", label="预测值", color="#A23B72", markersize=8, linewidth=2)

            # 置信区间阴影
            ax.fill_between(
                fc["projected_date"],
                fc["lower_95ci"],
                fc["upper_95ci"],
                alpha=0.3,
                color="#A23B72",
                label="95% 置信区间"
            )

            # 节假日标记
            holiday_fc = fc[fc["is_holiday_week"] == 1]
            if not holiday_fc.empty:
                ax.scatter(holiday_fc["projected_date"], holiday_fc["predicted_invitations"],
                          marker="*", s=300, color="red", zorder=5, label="节假日周")

        ax.set_title(f"{stream}", fontsize=12, weight="bold")
        ax.set_xlabel("日期", fontsize=10)
        ax.set_ylabel("邀请数", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(len(streams), 4):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = FIGURES_DIR / "forecast_with_intervals.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_feature_importance(df: pd.DataFrame) -> Path:
    """特征重要性分析（基于Random Forest）。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.enhanced_modeling import add_enhanced_features

    feats = add_enhanced_features(df)
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

    # 汇总平均重要性
    avg_imp = imp_df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(12)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_imp.values, y=avg_imp.index, palette="viridis")
    plt.title("特征重要性分析（Random Forest 平均）", fontsize=14, weight="bold")
    plt.xlabel("重要性得分", fontsize=12)
    plt.ylabel("特征", fontsize=12)
    plt.tight_layout()

    output_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_model_performance_comparison() -> Path:
    """模型性能对比图。"""
    # 读取报告数据
    try:
        with open("reports/enhanced_model_report.md", "r", encoding="utf-8") as f:
            content = f.read()

        # 解析表格（简化版，实际应用可用更健壮的解析）
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
                    metrics.append({
                        "stream": parts[0],
                        "model": parts[1],
                        "mae": float(parts[2])
                    })
            elif in_table and not line.startswith("|"):
                break

        if not metrics:
            return FIGURES_DIR / "model_comparison.png"

        df = pd.DataFrame(metrics)

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))

        # 分组柱状图
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

        ax.set_xlabel("流类型", fontsize=12, weight="bold")
        ax.set_ylabel("MAE（越低越好）", fontsize=12, weight="bold")
        ax.set_title("模型性能对比（滚动时间序列验证）", fontsize=14, weight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(streams, rotation=15, ha="right")
        ax.legend(title="模型", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = FIGURES_DIR / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path

    except Exception as e:
        print(f"⚠️  生成模型对比图失败: {e}")
        return FIGURES_DIR / "model_comparison.png"


def plot_stream_trends_combined(df: pd.DataFrame, forecast_df: pd.DataFrame) -> Path:
    """综合流趋势图（历史+预测+置信区间）。"""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))

    # 主要流选择
    main_streams = [
        "Alberta Express Entry Stream",
        "Alberta Opportunity Stream",
        "Dedicated Health Care Pathway"
    ]

    colors = {"Alberta Express Entry Stream": "#2E86AB",
              "Alberta Opportunity Stream": "#A23B72",
              "Dedicated Health Care Pathway": "#F18F01"}

    for stream in main_streams:
        # 历史数据
        hist = df[df["stream"] == stream].dropna(subset=["invitations"]).sort_values("draw_date")
        if not hist.empty:
            ax.plot(hist["draw_date"], hist["invitations"],
                   "o-", label=f"{stream} (历史)",
                   color=colors.get(stream, "#666"), alpha=0.7, linewidth=2, markersize=5)

        # 预测数据
        if not forecast_df.empty:
            fc = forecast_df[forecast_df["stream"] == stream].sort_values("projected_date")
            if not fc.empty:
                ax.plot(fc["projected_date"], fc["predicted_invitations"],
                       "s--", label=f"{stream} (预测)",
                       color=colors.get(stream, "#666"), linewidth=2, markersize=7)

                # 置信区间
                ax.fill_between(
                    fc["projected_date"],
                    fc["lower_95ci"],
                    fc["upper_95ci"],
                    alpha=0.15,
                    color=colors.get(stream, "#666")
                )

    ax.set_title("AAIP 主要流抽签趋势与预测（2025）", fontsize=16, weight="bold")
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("邀请数", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / "stream_trends_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_residual_diagnostics(df: pd.DataFrame) -> Path:
    """残差诊断图。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.enhanced_modeling import add_enhanced_features

    feats = add_enhanced_features(df)
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

        # 训练模型
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(g[feature_cols], g["invitations"])
        predictions = model.predict(g[feature_cols])
        residuals = g["invitations"].values - predictions

        residuals_all.extend(residuals)

        # 子图1: 残差 vs 预测值
        axes[0, 0].scatter(predictions, residuals, alpha=0.6, s=50, label=stream)

    if residuals_all:
        # 子图1设置
        axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel("预测值", fontsize=11)
        axes[0, 0].set_ylabel("残差", fontsize=11)
        axes[0, 0].set_title("残差 vs 预测值", fontsize=12, weight="bold")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        # 子图2: 残差直方图
        axes[0, 1].hist(residuals_all, bins=30, color="#2E86AB", alpha=0.7, edgecolor="black")
        axes[0, 1].axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("残差", fontsize=11)
        axes[0, 1].set_ylabel("频数", fontsize=11)
        axes[0, 1].set_title("残差分布", fontsize=12, weight="bold")
        axes[0, 1].grid(alpha=0.3)

        # 子图3: Q-Q图
        from scipy import stats
        stats.probplot(residuals_all, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q 图（正态性检验）", fontsize=12, weight="bold")
        axes[1, 0].grid(alpha=0.3)

        # 子图4: 残差时间序列
        axes[1, 1].plot(residuals_all, "o-", alpha=0.6, color="#A23B72", markersize=4)
        axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[1, 1].set_xlabel("观测序号", fontsize=11)
        axes[1, 1].set_ylabel("残差", fontsize=11)
        axes[1, 1].set_title("残差时间序列", fontsize=12, weight="bold")
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "residual_diagnostics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def run() -> Dict[str, Path]:
    """生成所有增强版可视化。"""
    df, forecast_df = load_data()

    outputs = {}

    print("📊 生成预测区间图...")
    outputs["forecast_intervals"] = plot_forecast_with_intervals(df, forecast_df)

    print("📊 生成特征重要性图...")
    outputs["feature_importance"] = plot_feature_importance(df)

    print("📊 生成模型性能对比图...")
    outputs["model_comparison"] = plot_model_performance_comparison()

    print("📊 生成综合趋势图...")
    outputs["stream_trends"] = plot_stream_trends_combined(df, forecast_df)

    print("📊 生成残差诊断图...")
    outputs["residual_diagnostics"] = plot_residual_diagnostics(df)

    return outputs


if __name__ == "__main__":
    results = run()
    print("\n✅ 所有图表已生成：")
    for name, path in results.items():
        print(f"  - {name}: {path}")
