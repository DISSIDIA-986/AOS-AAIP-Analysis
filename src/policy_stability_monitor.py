"""
政策稳定性监控模块
-------------------
目的：检测AAIP抽签模式的结构性变化，识别政策调整信号。
适用场景：2025年政策相对稳定，需监控未来政策变化以决定是否扩展训练数据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


def detect_concept_drift(
    df: pd.DataFrame,
    baseline_months: List[str] = ["2025-02", "2025-03", "2025-04"],
    test_months: List[str] = ["2025-10", "2025-11"],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    检测抽签模式的分布漂移（Concept Drift）。

    方法：Kolmogorov-Smirnov检验比较基线期与测试期的邀请数分布。

    Args:
        df: 清洗后的抽签数据
        baseline_months: 基线期月份（政策稳定早期）
        test_months: 测试期月份（最近月份）
        alpha: 显著性水平

    Returns:
        各流的KS检验结果 {"stream": {"statistic": float, "p_value": float, "drift_detected": bool}}
    """
    results = {}

    for stream, g in df.dropna(subset=["invitations"]).groupby("stream"):
        if len(g) < 10:
            continue

        # 分割基线期和测试期
        baseline = g[g["draw_date"].dt.to_period("M").astype(str).isin(baseline_months)]["invitations"]
        test = g[g["draw_date"].dt.to_period("M").astype(str).isin(test_months)]["invitations"]

        if len(baseline) < 3 or len(test) < 3:
            continue

        # KS检验：检测分布是否显著不同
        stat, p_value = ks_2samp(baseline, test)

        results[stream] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": p_value < alpha,
            "baseline_mean": round(float(baseline.mean()), 2),
            "test_mean": round(float(test.mean()), 2),
            "mean_change_pct": round(float((test.mean() - baseline.mean()) / baseline.mean() * 100), 2),
        }

    return results


def detect_variance_change(
    df: pd.DataFrame,
    window_size: int = 10,
    threshold_multiplier: float = 2.0,
) -> Dict[str, List[str]]:
    """
    检测邀请数方差的突变（异常波动信号）。

    方法：滚动窗口标准差，检测超过阈值的异常波动期。

    Args:
        window_size: 滚动窗口大小（抽签次数）
        threshold_multiplier: 异常阈值倍数（标准差的倍数）

    Returns:
        各流的异常波动日期列表
    """
    anomalies = {}

    for stream, g in df.dropna(subset=["invitations"]).groupby("stream"):
        if len(g) < window_size + 5:
            continue

        g = g.sort_values("draw_date").copy()
        rolling_std = g["invitations"].rolling(window=window_size, min_periods=5).std()
        threshold = rolling_std.mean() + threshold_multiplier * rolling_std.std()

        anomaly_dates = g.loc[rolling_std > threshold, "draw_date"].dt.date.astype(str).tolist()

        if anomaly_dates:
            anomalies[stream] = anomaly_dates

    return anomalies


def generate_stability_report(df: pd.DataFrame, output_path: Path) -> None:
    """生成政策稳定性监控报告。"""
    drift_results = detect_concept_drift(df)
    variance_anomalies = detect_variance_change(df)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# AAIP 政策稳定性监控报告\n\n")
        f.write("**目的**: 检测2025年AAIP抽签模式的结构性变化，识别政策调整信号。\n\n")

        f.write("## 1. 分布漂移检测（Concept Drift）\n\n")
        f.write("**方法**: Kolmogorov-Smirnov检验比较2025年早期（2-4月）vs 晚期（10-11月）邀请数分布。\n\n")

        if drift_results:
            f.write("| Stream | KS统计量 | p-value | 漂移检测 | 基线均值 | 测试均值 | 变化% |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for stream, res in drift_results.items():
                drift_flag = "⚠️ 是" if res["drift_detected"] else "✅ 否"
                f.write(
                    f"| {stream} | {res['ks_statistic']} | {res['p_value']} | "
                    f"{drift_flag} | {res['baseline_mean']} | {res['test_mean']} | {res['mean_change_pct']}% |\n"
                )
            f.write("\n**解读**:\n")
            f.write("- p-value < 0.05 → 检测到显著分布漂移，可能存在政策调整\n")
            f.write("- p-value ≥ 0.05 → 分布稳定，2025年政策同质性良好\n\n")
        else:
            f.write("无足够数据进行漂移检测。\n\n")

        f.write("## 2. 方差突变检测\n\n")
        f.write("**方法**: 滚动窗口标准差，检测超过正常波动2倍的异常波动期。\n\n")

        if variance_anomalies:
            for stream, dates in variance_anomalies.items():
                f.write(f"### {stream}\n")
                f.write(f"异常波动日期: {', '.join(dates)}\n\n")
        else:
            f.write("✅ 未检测到异常波动，各流邀请数方差稳定。\n\n")

        f.write("## 3. 数据扩展建议\n\n")

        drift_count = sum(1 for res in drift_results.values() if res["drift_detected"])

        if drift_count == 0:
            f.write("✅ **建议**: 2025年政策稳定，可安全使用2025全年数据训练模型。\n\n")
            f.write("**扩展策略**: 仅在2026年前2个月数据通过同质性检验后，再考虑扩展训练集。\n\n")
        elif drift_count <= len(drift_results) / 2:
            f.write("⚠️ **建议**: 部分流检测到漂移，建议分流建模或添加时间趋势特征。\n\n")
        else:
            f.write("🚨 **警告**: 多数流检测到显著漂移，可能存在政策调整。\n\n")
            f.write("**建议**: 使用最近3-6个月数据重新训练，避免使用整个2025年数据。\n\n")

        f.write("---\n\n")
        f.write("**生成时间**: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + "\n")


def run() -> None:
    """执行政策稳定性监控。"""
    df = pd.read_csv("data/processed/aaip_draws_2025.csv", parse_dates=["draw_date"])
    output_path = Path("reports/policy_stability_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_stability_report(df, output_path)
    print(f"政策稳定性报告已生成 -> {output_path}")


if __name__ == "__main__":
    run()
