import os
from dataclasses import dataclass


@dataclass
class Config:
    PROJECT_NAME: str = "WCA_GRU_RUL"
    SEED: int = 42
    DEVICE: str = "cuda"

    # Default paths (relative to repository root)
    DATA_DIR: str = "data/raw"
    OUTPUT_DIR: str = "data/processed"
    MODEL_DIR: str = "outputs/models"
    LOG_DIR: str = "outputs/logs"

    # =========================
    # Modality & Ablation Mode
    # =========================
    # Options:
    # "wca_gru"   : Proposed multimodal Cross-Attention model
    # "vib_only"  : Vibration-only baseline (Case 1 in Paper)
    # "temp_only" : Temperature-only baseline (Case 2 in Paper)
    # "concat"    : Early fusion concatenation (Case 3 in Paper)
    MODALITY_MODE: str = "wca_gru"

    # =========================
    # Dataset settings
    # =========================
    # Jung et al. 2024 dataset contains vibration X/Y + bearing temp
    VIB_COLUMNS: tuple = ("vibration_x", "vibration_y")
    TEMP_COLUMNS: tuple = ("bearing_temperature",)

    TIME_COLUMN: str = "time"
    TARGET_COLUMN: str = "rul"

    # =========================
    # Sliding window settings
    # =========================
    WINDOW_SIZE: int = 2048
    STRIDE: int = 512
    SEQ_LEN: int = 16

    # =========================
    # Wavelet feature settings
    # =========================
    WAVELET_NAME: str = "db4"
    WAVELET_LEVEL: int = 4

    VIB_FEATURE_DIM: int = 64
    TEMP_FEATURE_DIM: int = 8

    # =========================
    # Model settings
    # =========================
    VIB_EMBED_DIM: int = 128
    TEMP_EMBED_DIM: int = 128
    FUSION_DIM: int = 128

    CNN_CHANNELS: tuple = (64, 128)
    CNN_KERNEL_SIZE: int = 3
    CNN_DROPOUT: float = 0.2

    ATTENTION_HEADS: int = 4
    ATTENTION_DROPOUT: float = 0.1

    # "gru", "lstm", or "bilstm"
    RNN_TYPE: str = "gru"
    RNN_HIDDEN_DIM: int = 128
    RNN_LAYERS: int = 2
    RNN_DROPOUT: float = 0.2

    FC_HIDDEN_DIM: int = 128
    FC_DROPOUT: float = 0.3

    # =========================
    # Training settings
    # =========================
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    EARLY_STOPPING_PATIENCE: int = 15
    GRAD_CLIP_NORM: float = 1.0

    # Loss options: "mse", "mae", "huber", "asymmetric"
    LOSS_NAME: str = "asymmetric"
    HUBER_DELTA: float = 1.0

    # =========================
    # Split settings
    # =========================
    # "S1", "S2", or "S3"
    SPLIT_MODE: str = "S3"

    # S1: Train 0-60%, Val 60-70%, Test 70-100%
    S1_TRAIN_END: float = 0.60
    S1_VAL_END: float = 0.70

    # S2: Train 0-70%, Val 70-80%, Test 80-100%
    S2_TRAIN_END: float = 0.70
    S2_VAL_END: float = 0.80

    # S3 rolling-origin
    ROLLING_ORIGINS: tuple = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    ROLLING_HORIZON: float = 0.10

    # =========================
    # Normalization
    # =========================
    NORMALIZE_FEATURES: bool = True
    TARGET_SCALE: str = "none"  # "none", "minmax"
    PIECEWISE_RUL: bool = True
    MAX_RUL: float = 60.0

    # =========================
    # Utilities
    # =========================
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True

    @staticmethod
    def make_dirs():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)


cfg = Config()
