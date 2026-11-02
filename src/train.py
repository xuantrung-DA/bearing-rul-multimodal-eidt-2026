import os
import json
import random
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from config import cfg
    from model import build_model, count_parameters
    from evaluate import (
        evaluate_model_on_loader,
        predict_model,
        evaluate_predictions,
        print_metrics,
    )
except ImportError:
    from .config import cfg
    from .model import build_model, count_parameters
    from .evaluate import (
        evaluate_model_on_loader,
        predict_model,
        evaluate_predictions,
        print_metrics,
    )


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PyTorch Dataset
# ============================================================

class RULDataset(Dataset):
    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"File not found: {npz_path}")

        data = np.load(npz_path)
        self.X_vib = data["X_vib"].astype(np.float32)
        self.X_temp = data["X_temp"].astype(np.float32)
        self.y = data["y"].astype(np.float32)

        assert len(self.X_vib) == len(self.X_temp) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "X_vib": torch.tensor(self.X_vib[idx], dtype=torch.float32),
            "X_temp": torch.tensor(self.X_temp[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def make_dataloader(npz_path, batch_size, shuffle):
    dataset = RULDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
    )


# ============================================================
# Loss Functions
# ============================================================

class AsymmetricLoss(nn.Module):
    def __init__(self, late_penalty=5.0):
        super().__init__()
        self.late_penalty = late_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, true):
        loss = self.mse(pred, true)
        late_mask = pred > true
        loss[late_mask] = loss[late_mask] * self.late_penalty
        return loss.mean()


def build_loss():
    loss_name = cfg.LOSS_NAME.lower()

    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=cfg.HUBER_DELTA)
    if loss_name == "asymmetric":
        return AsymmetricLoss(late_penalty=5.0)

    raise ValueError(f"Unknown LOSS_NAME: {cfg.LOSS_NAME}")


# ============================================================
# Train & Validation Steps
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_count = 0.0, 0

    for batch in loader:
        X_vib = batch["X_vib"].to(device)
        X_temp = batch["X_temp"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        pred = model(X_vib, X_temp)
        loss = criterion(pred, y)
        loss.backward()

        if cfg.GRAD_CLIP_NORM is not None and cfg.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP_NORM)

        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            X_vib = batch["X_vib"].to(device)
            X_temp = batch["X_temp"].to(device)
            y = batch["y"].to(device)

            pred = model(X_vib, X_temp)
            loss = criterion(pred, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "config": cfg.__dict__,
        },
        path,
    )


def train_model(train_npz, val_npz, test_npz, split_name, run_output_dir):
    os.makedirs(run_output_dir, exist_ok=True)
    set_seed(cfg.SEED)

    device = torch.device("cuda" if cfg.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Training split: {split_name} | Device: {device}")
    print("=" * 80)

    train_loader = make_dataloader(train_npz, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_npz, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(test_npz, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = build_model(cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(run_output_dir, f"{split_name}_best_model.pt")

    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_mae": [], "lr": []}
    patience_counter = 0

    local_scaler_path = os.path.join(os.path.dirname(train_npz), "target_scaler.pkl")
    global_scaler_path = os.path.join(cfg.OUTPUT_DIR, "target_scaler.pkl")

    if cfg.TARGET_SCALE.lower() == "minmax":
        if os.path.exists(local_scaler_path):
            target_scaler = joblib.load(local_scaler_path)
        elif os.path.exists(global_scaler_path):
            target_scaler = joblib.load(global_scaler_path)
        else:
            target_scaler = None
    else:
        target_scaler = None

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        y_val_true, y_val_pred = predict_model(model, val_loader, device)
        val_metrics, _, _, _ = evaluate_predictions(y_val_true, y_val_pred, target_scaler=target_scaler)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_rmse"].append(float(val_metrics["RMSE"]))
        history["val_mae"].append(float(val_metrics["MAE"]))
        history["lr"].append(float(current_lr))

        print(
            f"Epoch [{epoch:03d}/{cfg.EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSE: {val_metrics['RMSE']:.4f} | "
            f"Val MAE: {val_metrics['MAE']:.4f} | "
            f"LR: {current_lr:.6e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val_loss, history)
            print(f"  -> Saved best model: {best_ckpt_path}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    history_path = os.path.join(run_output_dir, f"{split_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print("\nLoading best checkpoint for final evaluation...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_stage_metrics, _, _ = evaluate_model_on_loader(
        model, val_loader, device, target_scaler=target_scaler, output_dir=run_output_dir, split_name=f"{split_name}_val"
    )

    test_metrics, test_stage_metrics, _, _ = evaluate_model_on_loader(
        model, test_loader, device, target_scaler=target_scaler, output_dir=run_output_dir, split_name=f"{split_name}_test"
    )

    summary = {
        "split_name": split_name,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(checkpoint["epoch"]),
        "val_metrics": val_metrics,
        "val_stage_metrics": val_stage_metrics,
        "test_metrics": test_metrics,
        "test_stage_metrics": test_stage_metrics,
        "best_checkpoint": best_ckpt_path,
    }

    summary_path = os.path.join(run_output_dir, f"{split_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def run_s1_or_s2():
    split_mode = cfg.SPLIT_MODE.upper()
    train_npz = os.path.join(cfg.OUTPUT_DIR, f"{split_mode}_train.npz")
    val_npz = os.path.join(cfg.OUTPUT_DIR, f"{split_mode}_val.npz")
    test_npz = os.path.join(cfg.OUTPUT_DIR, f"{split_mode}_test.npz")
    run_output_dir = os.path.join(cfg.OUTPUT_DIR, "training_runs", split_mode)

    return train_model(train_npz, val_npz, test_npz, split_mode, run_output_dir)


def find_s3_origin_dirs():
    origin_dirs = []
    if os.path.exists(cfg.OUTPUT_DIR):
        for name in os.listdir(cfg.OUTPUT_DIR):
            path = os.path.join(cfg.OUTPUT_DIR, name)
            if os.path.isdir(path) and name.startswith("S3_origin_"):
                origin_dirs.append(path)
    return sorted(origin_dirs)


def run_s3():
    origin_dirs = find_s3_origin_dirs()
    if len(origin_dirs) == 0:
        raise FileNotFoundError(f"No S3_origin_* folders found in {cfg.OUTPUT_DIR}. Run preprocess.py with SPLIT_MODE='S3' first.")

    all_summaries = []
    for origin_dir in origin_dirs:
        origin_name = os.path.basename(origin_dir)
        train_npz = os.path.join(origin_dir, "train.npz")
        val_npz = os.path.join(origin_dir, "val.npz")
        test_npz = os.path.join(origin_dir, "test.npz")
        run_output_dir = os.path.join(cfg.OUTPUT_DIR, "training_runs", "S3", origin_name)

        summary = train_model(train_npz, val_npz, test_npz, origin_name, run_output_dir)
        all_summaries.append(summary)

    s3_summary_path = os.path.join(cfg.OUTPUT_DIR, "training_runs", "S3", "S3_all_summaries.json")
    os.makedirs(os.path.dirname(s3_summary_path), exist_ok=True)
    with open(s3_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4)

    return all_summaries


def main():
    cfg.make_dirs()
    split_mode = cfg.SPLIT_MODE.upper()

    if split_mode in ["S1", "S2"]:
        run_s1_or_s2()
    elif split_mode == "S3":
        run_s3()
    else:
        raise ValueError(f"Invalid SPLIT_MODE: {cfg.SPLIT_MODE}. Use 'S1', 'S2', or 'S3'.")


if __name__ == "__main__":
    main()
