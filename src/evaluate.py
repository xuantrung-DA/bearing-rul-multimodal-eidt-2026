import os
import json
import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Basic signal / feature metrics
# ============================================================

def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2)))


def mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def peak_to_peak(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.ptp(x))


def skewness(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x) + eps
    return float(np.mean(((x - mean) / std) ** 3))


def kurtosis(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x) + eps
    return float(np.mean(((x - mean) / std) ** 4))


def crest_factor(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x)) / (rms(x) + eps))


def impulse_factor(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x)) / (np.mean(np.abs(x)) + eps))


def margin_factor(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x)) / ((np.mean(np.sqrt(np.abs(x))) ** 2) + eps))


def shape_factor(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    return float(rms(x) / (np.mean(np.abs(x)) + eps))


def signal_statistics(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": rms(x),
        "mse": float(np.mean(x ** 2)),
        "peak_to_peak": peak_to_peak(x),
        "skewness": skewness(x),
        "kurtosis": kurtosis(x),
        "crest_factor": crest_factor(x),
        "impulse_factor": impulse_factor(x),
        "margin_factor": margin_factor(x),
        "shape_factor": shape_factor(x),
        "energy": float(np.sum(x ** 2)),
    }


# ============================================================
# RUL-specific metrics
# ============================================================

def phm_score(y_true, y_pred):
    """
    PHM 2008 asymmetric penalty score.
    d = y_pred - y_true
    Late predictions (d > 0) are penalized more heavily.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    d = y_pred - y_true
    score = 0.0

    for e in d:
        if e < 0:
            score += math.exp(-e / 13.0) - 1.0
        else:
            score += math.exp(e / 10.0) - 1.0

    return float(score / len(d))


def late_prediction_rate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_pred > y_true))


def early_prediction_rate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_pred < y_true))


def early_late_error_ratio(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_pred - y_true

    early_errors = np.abs(error[error < 0])
    late_errors = np.abs(error[error > 0])

    early_mean = np.mean(early_errors) if len(early_errors) > 0 else 0.0
    late_mean = np.mean(late_errors) if len(late_errors) > 0 else 0.0

    return float(late_mean / (early_mean + eps))


def normalized_rmse(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true) + eps))


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "NRMSE": normalized_rmse(y_true, y_pred),
        "PHM_SCORE": phm_score(y_true, y_pred),
        "LATE_RATE": late_prediction_rate(y_true, y_pred),
        "EARLY_RATE": early_prediction_rate(y_true, y_pred),
        "EARLY_LATE_RATIO": early_late_error_ratio(y_true, y_pred),
    }


# ============================================================
# Stage-wise metrics (Healthy, Degradation, Near-failure)
# ============================================================

def get_life_stage_from_rul(y_true):
    y_true = np.asarray(y_true, dtype=np.float64)
    max_rul = np.max(y_true)
    progress = 1.0 - (y_true / (max_rul + 1e-8))

    stages = []
    for p in progress:
        if p < 0.4:
            stages.append("healthy")
        elif p < 0.8:
            stages.append("degradation")
        else:
            stages.append("near_failure")

    return np.asarray(stages)


def compute_stage_metrics(y_true, y_pred):
    stages = get_life_stage_from_rul(y_true)
    results = {}

    for stage in ["healthy", "degradation", "near_failure"]:
        mask = stages == stage
        if np.sum(mask) == 0:
            continue

        results[stage] = compute_regression_metrics(
            y_true=np.asarray(y_true)[mask],
            y_pred=np.asarray(y_pred)[mask],
        )
        results[stage]["N"] = int(np.sum(mask))

    return results


# ============================================================
# Inference & Reporting Helpers
# ============================================================

def predict_model(model, dataloader, device):
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in dataloader:
            X_vib = batch["X_vib"].to(device)
            X_temp = batch["X_temp"].to(device)
            y = batch["y"].to(device)

            pred = model(X_vib, X_temp)
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())

    return np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0)


def inverse_transform_target(y, target_scaler=None):
    y = np.asarray(y).reshape(-1, 1)
    if target_scaler is None:
        return y.reshape(-1)
    return target_scaler.inverse_transform(y).reshape(-1)


def evaluate_predictions(y_true, y_pred, target_scaler=None):
    y_true_inv = inverse_transform_target(y_true, target_scaler)
    y_pred_inv = inverse_transform_target(y_pred, target_scaler)

    overall_metrics = compute_regression_metrics(y_true_inv, y_pred_inv)
    stage_metrics = compute_stage_metrics(y_true_inv, y_pred_inv)

    return overall_metrics, stage_metrics, y_true_inv, y_pred_inv


def print_metrics(metrics, title="Metrics"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.6f}")
        else:
            print(f"{k:20s}: {v}")


def print_stage_metrics(stage_metrics):
    print("\n" + "=" * 80)
    print("Stage-wise Metrics")
    print("=" * 80)
    for stage, metrics in stage_metrics.items():
        print(f"\n[{stage}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:20s}: {v:.6f}")
            else:
                print(f"{k:20s}: {v}")


def save_evaluation_report(
    output_dir,
    split_name,
    overall_metrics,
    stage_metrics,
    y_true,
    y_pred,
):
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "split": split_name,
        "overall_metrics": overall_metrics,
        "stage_metrics": stage_metrics,
    }

    report_path = os.path.join(output_dir, f"{split_name}_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "error": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true),
    })

    pred_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    print(f"\nSaved report     : {report_path}")
    print(f"Saved predictions: {pred_path}")


def evaluate_model_on_loader(
    model,
    dataloader,
    device,
    target_scaler=None,
    output_dir=None,
    split_name="test",
):
    y_true, y_pred = predict_model(model, dataloader, device)
    overall_metrics, stage_metrics, y_true_inv, y_pred_inv = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        target_scaler=target_scaler,
    )

    print_metrics(overall_metrics, title=f"{split_name.upper()} Overall Metrics")
    print_stage_metrics(stage_metrics)

    if output_dir is not None:
        save_evaluation_report(
            output_dir=output_dir,
            split_name=split_name,
            overall_metrics=overall_metrics,
            stage_metrics=stage_metrics,
            y_true=y_true_inv,
            y_pred=y_pred_inv,
        )

    return overall_metrics, stage_metrics, y_true_inv, y_pred_inv
