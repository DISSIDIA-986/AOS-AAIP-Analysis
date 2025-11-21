"""
在线学习更新机制
---------------
目的：随着新抽签数据产生，增量更新模型而非完全重新训练。
适用场景：2025年12月数据陆续产生，需实时捕捉最新趋势。

方法：
1. 滑动窗口：保留最近N次抽签作为训练集
2. 增量更新：添加新数据后重新训练（树模型需完全重训，但数据窗口限制了计算量）
3. 性能监控：跟踪模型在新数据上的MAE，检测性能退化
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/aaip_draws_2025.csv")
MODEL_REGISTRY = Path("models/model_registry.json")
PERFORMANCE_LOG = Path("models/performance_log.csv")


def load_model_registry() -> Dict[str, Dict]:
    """加载模型注册表（存储每个流的最优模型配置）。"""
    if MODEL_REGISTRY.exists():
        with open(MODEL_REGISTRY, "r") as f:
            return json.load(f)
    return {}


def save_model_registry(registry: Dict[str, Dict]) -> None:
    """保存模型注册表。"""
    MODEL_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)


def incremental_update(
    stream: str,
    new_data: pd.DataFrame,
    window_size: int = 50,
    feature_cols: List[str] = None,
) -> Dict[str, object]:
    """
    增量更新单个流的模型。

    Args:
        stream: 流名称
        new_data: 新增的抽签数据
        window_size: 滑动窗口大小（保留最近N次抽签）
        feature_cols: 特征列表

    Returns:
        更新结果字典（包含新MAE、数据量等）
    """
    if feature_cols is None:
        feature_cols = [
            "date_ord", "gap_days", "lag1_inv", "lag2_inv", "lag3_inv",
            "roll3_inv", "roll5_inv", "lag1_score", "sin_doy", "cos_doy",
            "month_num", "event_index", "is_holiday_week", "is_priority_sector",
            "cumulative_invitations", "gap_deviation", "is_gap_anomaly"
        ]

    # 加载注册表
    registry = load_model_registry()
    stream_config = registry.get(stream, {"model_type": "RandomForest"})

    # 准备训练数据（滑动窗口）
    stream_data = new_data[new_data["stream"] == stream].dropna(subset=["invitations"]).sort_values("draw_date")

    if len(stream_data) < 15:
        return {
            "stream": stream,
            "status": "insufficient_data",
            "data_points": len(stream_data),
        }

    # 取最近window_size条记录
    train_data = stream_data.tail(window_size).copy()
    train_data = train_data.dropna(subset=feature_cols + ["invitations"])

    if train_data.empty or len(train_data) < 10:
        return {
            "stream": stream,
            "status": "insufficient_valid_data",
            "data_points": len(train_data),
        }

    # 训练测试分割（最后3次作为验证）
    if len(train_data) < 13:
        return {
            "stream": stream,
            "status": "insufficient_for_validation",
            "data_points": len(train_data),
        }

    train, test = train_data.iloc[:-3], train_data.iloc[-3:]

    # 训练模型
    if stream_config["model_type"] == "RandomForest":
        model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    elif stream_config["model_type"] == "GradientBoosting":
        model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    model.fit(train[feature_cols], train["invitations"])

    # 评估
    predictions = model.predict(test[feature_cols])
    mae = mean_absolute_error(test["invitations"], predictions)

    # 更新注册表
    stream_config.update({
        "model_type": stream_config["model_type"],
        "last_updated": datetime.now().isoformat(),
        "training_samples": len(train),
        "validation_mae": round(float(mae), 2),
        "window_size": window_size,
    })
    registry[stream] = stream_config
    save_model_registry(registry)

    # 记录性能日志
    log_performance(stream, mae, len(train), len(test))

    return {
        "stream": stream,
        "status": "updated",
        "mae": round(float(mae), 2),
        "training_samples": len(train),
        "test_samples": len(test),
        "model_type": stream_config["model_type"],
    }


def log_performance(stream: str, mae: float, train_size: int, test_size: int) -> None:
    """记录模型性能到日志文件。"""
    PERFORMANCE_LOG.parent.mkdir(parents=True, exist_ok=True)

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "stream": stream,
        "mae": mae,
        "train_size": train_size,
        "test_size": test_size,
    }])

    if PERFORMANCE_LOG.exists():
        existing = pd.read_csv(PERFORMANCE_LOG)
        log_entry = pd.concat([existing, log_entry], ignore_index=True)

    log_entry.to_csv(PERFORMANCE_LOG, index=False)


def detect_performance_degradation(stream: str, threshold: float = 1.2) -> Dict[str, object]:
    """
    检测模型性能退化。

    Args:
        stream: 流名称
        threshold: 退化阈值（当前MAE / 历史最佳MAE）

    Returns:
        退化检测结果
    """
    if not PERFORMANCE_LOG.exists():
        return {"stream": stream, "degradation_detected": False, "reason": "no_history"}

    log = pd.read_csv(PERFORMANCE_LOG)
    stream_log = log[log["stream"] == stream]

    if len(stream_log) < 2:
        return {"stream": stream, "degradation_detected": False, "reason": "insufficient_history"}

    historical_best = stream_log["mae"].min()
    current_mae = stream_log.iloc[-1]["mae"]

    ratio = current_mae / historical_best

    return {
        "stream": stream,
        "degradation_detected": ratio > threshold,
        "current_mae": round(float(current_mae), 2),
        "historical_best_mae": round(float(historical_best), 2),
        "degradation_ratio": round(float(ratio), 2),
        "threshold": threshold,
    }


def batch_update_all_streams(data_path: Path = DATA_PATH) -> List[Dict[str, object]]:
    """批量更新所有流的模型。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.enhanced_modeling import add_enhanced_features

    df = pd.read_csv(data_path, parse_dates=["draw_date"])
    feats = add_enhanced_features(df)

    results = []
    for stream in feats["stream"].unique():
        result = incremental_update(stream, feats)
        results.append(result)

        # 检测退化
        degradation = detect_performance_degradation(stream)
        if degradation["degradation_detected"]:
            print(f"⚠️  {stream}: 性能退化检测！MAE从{degradation['historical_best_mae']}升至{degradation['current_mae']}")

    return results


def generate_update_report(results: List[Dict[str, object]], output_path: Path) -> None:
    """生成更新报告。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 在线学习更新报告\n\n")
        f.write(f"**更新时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 更新结果\n\n")
        f.write("| Stream | Status | MAE | Training Samples | Model Type |\n")
        f.write("| --- | --- | --- | --- | --- |\n")

        for res in results:
            stream = res.get("stream", "Unknown")
            status = res.get("status", "unknown")
            mae = res.get("mae", "N/A")
            train_samples = res.get("training_samples", "N/A")
            model_type = res.get("model_type", "N/A")

            status_icon = "✅" if status == "updated" else "⚠️"
            f.write(f"| {stream} | {status_icon} {status} | {mae} | {train_samples} | {model_type} |\n")

        f.write("\n## 性能退化检测\n\n")
        for res in results:
            stream = res.get("stream")
            degradation = detect_performance_degradation(stream)

            if degradation.get("degradation_detected"):
                f.write(f"### ⚠️ {stream}\n\n")
                f.write(f"- 当前MAE: {degradation['current_mae']}\n")
                f.write(f"- 历史最佳: {degradation['historical_best_mae']}\n")
                f.write(f"- 退化比例: {degradation['degradation_ratio']}x\n")
                f.write(f"- **建议**: 检查最新数据分布，可能需要调整特征或模型。\n\n")

        if not any(detect_performance_degradation(r["stream"]).get("degradation_detected") for r in results):
            f.write("✅ 所有流性能稳定，未检测到显著退化。\n\n")

        f.write("---\n\n")
        f.write("**下次更新**: 建议每周或新增5+次抽签后重新运行。\n")


def run() -> None:
    """执行在线学习更新。"""
    print("🔄 开始增量更新所有流的模型...")
    results = batch_update_all_streams()

    print("\n📊 更新结果：")
    for res in results:
        status_icon = "✅" if res.get("status") == "updated" else "⚠️"
        print(f"  {status_icon} {res['stream']}: {res.get('status')} (MAE: {res.get('mae', 'N/A')})")

    report_path = Path("reports/online_learning_update.md")
    generate_update_report(results, report_path)
    print(f"\n✅ 更新报告已生成 -> {report_path}")


if __name__ == "__main__":
    run()
