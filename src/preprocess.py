import os
import re
import glob
import json
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

try:
    from config import cfg
except ImportError:
    from .config import cfg


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def natural_sort_key(path: str):
    filename = os.path.basename(path)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


def find_csv_files(data_dir: str) -> List[str]:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    csv_files = sorted(csv_files, key=natural_sort_key)
    return csv_files


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = list(df.columns)
    rename_map = {}

    for col in original_cols:
        c = str(col).strip().lower()

        if c in [
            "vibration_x", "vib_x", "x", "x-axis", "x_axis",
            "horizontal vibration", "horizontal_vibration", "acc_x", "acceleration_x"
        ]:
            rename_map[col] = "vibration_x"
        elif c in [
            "vibration_y", "vib_y", "y", "y-axis", "y_axis",
            "vertical vibration", "vertical_vibration", "acc_y", "acceleration_y"
        ]:
            rename_map[col] = "vibration_y"
        elif c in [
            "bearing_temperature", "bearing temp", "bearing_temp",
            "temperature", "temp", "bearingtemperature"
        ]:
            rename_map[col] = "bearing_temperature"
        elif c in [
            "atmospheric_temperature", "atmospheric temp", "ambient_temperature",
            "ambient temp", "ambient_temp", "room_temperature", "room temp", "air_temperature"
        ]:
            rename_map[col] = "atmospheric_temperature"
        elif c in ["time", "timestamp", "t", "seconds", "sec"]:
            rename_map[col] = "time"

    return df.rename(columns=rename_map)


def ensure_required_columns(df: pd.DataFrame):
    required_cols = list(cfg.VIB_COLUMNS) + list(cfg.TEMP_COLUMNS)
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}. Existing: {list(df.columns)}")


# ============================================================
# Feature Extraction
# ============================================================

def safe_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "rms": 0.0, "ptp": 0.0, "skew": 0.0, "kurt": 0.0, "energy": 0.0
        }

    mean = np.mean(x)
    std = np.std(x) + 1e-8
    centered = x - mean
    skew = np.mean((centered / std) ** 3)
    kurt = np.mean((centered / std) ** 4)

    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "ptp": float(np.ptp(x)),
        "skew": float(skew),
        "kurt": float(kurt),
        "energy": float(np.sum(x ** 2) / len(x)),
    }


def wavelet_features_1d(
    signal: np.ndarray,
    wavelet_name: str = "db4",
    level: int = 4,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level, mode="symmetric")

    features = []
    for coeff in coeffs:
        stats = safe_stats(coeff)
        features.extend(list(stats.values()))

    return np.asarray(features, dtype=np.float32)


def fix_feature_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == target_dim:
        return x
    if len(x) > target_dim:
        return x[:target_dim]

    padded = np.zeros(target_dim, dtype=np.float32)
    padded[:len(x)] = x
    return padded


def vibration_window_features(window_df: pd.DataFrame) -> np.ndarray:
    feats = []
    for col in cfg.VIB_COLUMNS:
        sig = window_df[col].values.astype(np.float32)
        f = wavelet_features_1d(sig, wavelet_name=cfg.WAVELET_NAME, level=cfg.WAVELET_LEVEL)
        feats.append(f)

    feats = np.concatenate(feats, axis=0)
    return fix_feature_dim(feats, cfg.VIB_FEATURE_DIM)


def thermal_window_features(window_df: pd.DataFrame) -> np.ndarray:
    feats = []
    for col in cfg.TEMP_COLUMNS:
        x = window_df[col].values.astype(np.float32)
        slope = 0.0 if len(x) <= 1 else np.polyfit(np.arange(len(x), dtype=np.float32), x, deg=1)[0]

        mean = np.mean(x)
        std = np.std(x)
        min_v = np.min(x)
        max_v = np.max(x)
        first = x[0]
        last = x[-1]
        delta = last - first

        feats.extend([mean, std, min_v, max_v, first, last, delta, slope])

    feats = np.asarray(feats, dtype=np.float32)
    return fix_feature_dim(feats, cfg.TEMP_FEATURE_DIM)


# ============================================================
# Windowing & Sequence Assembly
# ============================================================

def create_windows_from_file(
    df: pd.DataFrame,
    hour_index: int,
    total_hours: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensure_required_columns(df)

    n = len(df)
    window_size = cfg.WINDOW_SIZE
    stride = cfg.STRIDE

    vib_features, temp_features, labels = [], [], []

    if n < window_size:
        return (
            np.empty((0, cfg.VIB_FEATURE_DIM), dtype=np.float32),
            np.empty((0, cfg.TEMP_FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    linear_rul = float(total_hours - 1 - hour_index)
    rul_value = min(linear_rul, float(cfg.MAX_RUL)) if getattr(cfg, "PIECEWISE_RUL", False) else linear_rul

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window_df = df.iloc[start:end]

        vib_f = vibration_window_features(window_df)
        temp_f = thermal_window_features(window_df)

        vib_features.append(vib_f)
        temp_features.append(temp_f)
        labels.append(rul_value)

    return (
        np.asarray(vib_features, dtype=np.float32),
        np.asarray(temp_features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def make_sequences(
    vib_windows: np.ndarray,
    temp_windows: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_windows = len(labels)

    if num_windows < seq_len:
        return (
            np.empty((0, seq_len, cfg.VIB_FEATURE_DIM), dtype=np.float32),
            np.empty((0, seq_len, cfg.TEMP_FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    seq_vib, seq_temp, seq_y = [], [], []

    for i in range(0, num_windows - seq_len + 1):
        j = i + seq_len
        seq_vib.append(vib_windows[i:j])
        seq_temp.append(temp_windows[i:j])
        seq_y.append(labels[j - 1])

    return (
        np.asarray(seq_vib, dtype=np.float32),
        np.asarray(seq_temp, dtype=np.float32),
        np.asarray(seq_y, dtype=np.float32),
    )


# ============================================================
# Temporal Splitting Strategies
# ============================================================

def get_split_indices(total_files: int, split_mode: str = "S1") -> Dict[str, List[int]]:
    indices = list(range(total_files))
    split_mode = split_mode.upper()

    if split_mode == "S1":
        train_end = int(total_files * cfg.S1_TRAIN_END)
        val_end = int(total_files * cfg.S1_VAL_END)
    elif split_mode == "S2":
        train_end = int(total_files * cfg.S2_TRAIN_END)
        val_end = int(total_files * cfg.S2_VAL_END)
    else:
        raise ValueError(f"S1 or S2 supported here. Got: {split_mode}")

    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def get_rolling_splits(total_files: int) -> List[Dict[str, List[int]]]:
    indices = list(range(total_files))
    splits = []

    for origin in cfg.ROLLING_ORIGINS:
        train_end = int(total_files * origin)
        test_end = int(total_files * min(origin + cfg.ROLLING_HORIZON, 1.0))
        if train_end <= 0 or test_end <= train_end:
            continue

        splits.append({
            "origin": origin,
            "train": indices[:train_end],
            "test": indices[train_end:test_end],
        })

    return splits


def process_file_list(
    csv_files: List[str],
    selected_indices: List[int],
    total_files: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_seq_vib, all_seq_temp, all_seq_y = [], [], []

    for idx in selected_indices:
        file_path = csv_files[idx]
        df = pd.read_csv(
            file_path,
            header=None,
            names=["vibration_x", "vibration_y", "bearing_temperature", "atmospheric_temperature"],
        )
        df = df.dropna().reset_index(drop=True)

        vib_w, temp_w, y_w = create_windows_from_file(df=df, hour_index=idx, total_hours=total_files)
        seq_vib, seq_temp, seq_y = make_sequences(
            vib_windows=vib_w, temp_windows=temp_w, labels=y_w, seq_len=cfg.SEQ_LEN
        )

        if len(seq_y) == 0:
            continue

        all_seq_vib.append(seq_vib)
        all_seq_temp.append(seq_temp)
        all_seq_y.append(seq_y)

    if len(all_seq_y) == 0:
        return (
            np.empty((0, cfg.SEQ_LEN, cfg.VIB_FEATURE_DIM), dtype=np.float32),
            np.empty((0, cfg.SEQ_LEN, cfg.TEMP_FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.concatenate(all_seq_vib, axis=0),
        np.concatenate(all_seq_temp, axis=0),
        np.concatenate(all_seq_y, axis=0),
    )


def normalize_datasets(train_data, val_data, test_data):
    Xv_train, Xt_train, y_train = train_data
    Xv_val, Xt_val, y_val = val_data
    Xv_test, Xt_test, y_test = test_data

    vib_scaler = StandardScaler()
    temp_scaler = StandardScaler()

    B, T, Fv = Xv_train.shape
    _, _, Ft = Xt_train.shape

    vib_scaler.fit(Xv_train.reshape(-1, Fv))
    temp_scaler.fit(Xt_train.reshape(-1, Ft))

    def transform_x(X, scaler):
        B, T, F = X.shape
        X2 = np.clip(scaler.transform(X.reshape(-1, F)), -10.0, 10.0)
        return X2.reshape(B, T, F).astype(np.float32)

    Xv_train = transform_x(Xv_train, vib_scaler)
    Xv_val = transform_x(Xv_val, vib_scaler)
    Xv_test = transform_x(Xv_test, vib_scaler)

    Xt_train = transform_x(Xt_train, temp_scaler)
    Xt_val = transform_x(Xt_val, temp_scaler)
    Xt_test = transform_x(Xt_test, temp_scaler)

    target_scaler = None
    if cfg.TARGET_SCALE == "minmax":
        target_scaler = MinMaxScaler()
        target_scaler.fit(y_train.reshape(-1, 1))
        y_train = target_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
        y_val = target_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
        y_test = target_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    return (
        (Xv_train, Xt_train, y_train),
        (Xv_val, Xt_val, y_val),
        (Xv_test, Xt_test, y_test),
        vib_scaler,
        temp_scaler,
        target_scaler,
    )


def save_npz(path: str, X_vib, X_temp, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X_vib=X_vib, X_temp=X_temp, y=y)


def save_split_info(path: str, split_info: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4, ensure_ascii=False)


def preprocess_s1_s2():
    csv_files = find_csv_files(cfg.DATA_DIR)
    total_files = len(csv_files)
    split_indices = get_split_indices(total_files=total_files, split_mode=cfg.SPLIT_MODE)

    train_data = process_file_list(csv_files, split_indices["train"], total_files)
    val_data = process_file_list(csv_files, split_indices["val"], total_files)
    test_data = process_file_list(csv_files, split_indices["test"], total_files)

    if cfg.NORMALIZE_FEATURES:
        (
            train_data, val_data, test_data,
            vib_scaler, temp_scaler, target_scaler
        ) = normalize_datasets(train_data, val_data, test_data)

        joblib.dump(vib_scaler, os.path.join(cfg.OUTPUT_DIR, "vib_scaler.pkl"))
        joblib.dump(temp_scaler, os.path.join(cfg.OUTPUT_DIR, "temp_scaler.pkl"))
        if target_scaler is not None:
            joblib.dump(target_scaler, os.path.join(cfg.OUTPUT_DIR, "target_scaler.pkl"))

    save_npz(os.path.join(cfg.OUTPUT_DIR, f"{cfg.SPLIT_MODE}_train.npz"), *train_data)
    save_npz(os.path.join(cfg.OUTPUT_DIR, f"{cfg.SPLIT_MODE}_val.npz"), *val_data)
    save_npz(os.path.join(cfg.OUTPUT_DIR, f"{cfg.SPLIT_MODE}_test.npz"), *test_data)

    split_info = {
        "split_mode": cfg.SPLIT_MODE,
        "total_files": total_files,
        "window_size": cfg.WINDOW_SIZE,
        "stride": cfg.STRIDE,
        "seq_len": cfg.SEQ_LEN,
        "vib_feature_dim": cfg.VIB_FEATURE_DIM,
        "temp_feature_dim": cfg.TEMP_FEATURE_DIM,
        "split_indices": split_indices,
        "csv_files": [os.path.basename(f) for f in csv_files],
    }
    save_split_info(os.path.join(cfg.OUTPUT_DIR, f"{cfg.SPLIT_MODE}_split_info.json"), split_info)


def preprocess_s3_rolling():
    csv_files = find_csv_files(cfg.DATA_DIR)
    total_files = len(csv_files)
    rolling_splits = get_rolling_splits(total_files)

    for i, split in enumerate(rolling_splits, start=1):
        origin = split["origin"]
        train_indices = split["train"]
        test_indices = split["test"]

        val_size = max(1, int(len(train_indices) * 0.1))
        real_train_indices = train_indices[:-val_size]
        val_indices = train_indices[-val_size:]

        train_data = process_file_list(csv_files, real_train_indices, total_files)
        val_data = process_file_list(csv_files, val_indices, total_files)
        test_data = process_file_list(csv_files, test_indices, total_files)

        split_dir = os.path.join(cfg.OUTPUT_DIR, f"S3_origin_{int(origin * 100)}")
        os.makedirs(split_dir, exist_ok=True)

        if cfg.NORMALIZE_FEATURES:
            (
                train_data, val_data, test_data,
                vib_scaler, temp_scaler, target_scaler
            ) = normalize_datasets(train_data, val_data, test_data)

            joblib.dump(vib_scaler, os.path.join(split_dir, "vib_scaler.pkl"))
            joblib.dump(temp_scaler, os.path.join(split_dir, "temp_scaler.pkl"))
            if target_scaler is not None:
                joblib.dump(target_scaler, os.path.join(split_dir, "target_scaler.pkl"))

        save_npz(os.path.join(split_dir, "train.npz"), *train_data)
        save_npz(os.path.join(split_dir, "val.npz"), *val_data)
        save_npz(os.path.join(split_dir, "test.npz"), *test_data)

        split_info = {
            "split_mode": "S3",
            "origin": origin,
            "total_files": total_files,
            "window_size": cfg.WINDOW_SIZE,
            "stride": cfg.STRIDE,
            "seq_len": cfg.SEQ_LEN,
            "vib_feature_dim": cfg.VIB_FEATURE_DIM,
            "temp_feature_dim": cfg.TEMP_FEATURE_DIM,
            "train_indices": real_train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "csv_files": [os.path.basename(f) for f in csv_files],
        }
        save_split_info(os.path.join(split_dir, "split_info.json"), split_info)


def main():
    set_seed(cfg.SEED)
    cfg.make_dirs()

    if cfg.SPLIT_MODE.upper() in ["S1", "S2"]:
        preprocess_s1_s2()
    elif cfg.SPLIT_MODE.upper() == "S3":
        preprocess_s3_rolling()
    else:
        raise ValueError(f"Invalid SPLIT_MODE: {cfg.SPLIT_MODE}. Use 'S1', 'S2', or 'S3'.")


if __name__ == "__main__":
    main()
