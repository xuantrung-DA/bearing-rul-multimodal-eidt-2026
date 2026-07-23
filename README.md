# Wavelet Cross-Attention GRU (WCA-GRU) for Bearing RUL Prediction

Official implementation of the paper:
**"A Multimodal Framework for Bearing Remaining Useful Life Prediction using Wavelet Cross-Attention GRU"**
*Authors:* Phan Anh Minh, Nguyen Xuan Trung, Phu-Nguyen Le.

---

## 📌 Overview

Remaining useful life (RUL) prediction of rolling-element bearings is critical in industrial predictive maintenance. While traditional deep learning models rely predominantly on high-frequency vibration signals, they often struggle to capture long-term, low-frequency degradation trends, particularly in the near-failure stage.

This framework (**WCA-GRU**) addresses these challenges by:
1. **Dual-Encoder Architecture:** Extracting time-frequency features from vibration signals via 1D-CNN over Wavelet coefficients, alongside low-frequency thermal features via a Multi-Layer Perceptron (MLP).
2. **Cross-Attention Fusion:** Dynamically recalibrating high-frequency vibration features using bearing temperature as degradation context.
3. **Leakage-Aware Evaluation:** Employing strict file-level forward temporal splits (**S1, S2**) and rolling-origin validation (**S3**) to avoid temporal data leakage.

---

## 📂 Repository Structure

```
WCA-GRU-Bearing-RUL/
├── src/
│   ├── config.py         # Hyperparameters, split settings, and model configs
│   ├── preprocess.py     # Wavelet feature extraction & leakage-aware splitting
│   ├── model.py          # WCA-GRU dual-encoder & cross-attention architecture
│   ├── evaluate.py       # Metrics (RMSE, MAE, PHM Score, Stage-wise breakdown)
│   └── train.py          # Training loop with early stopping & evaluation
├── .gitignore            # Excludes heavy datasets and binary checkpoints
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/WCA-GRU-Bearing-RUL.git
cd WCA-GRU-Bearing-RUL
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Setup
Download the **Ball Bearing Vibration and Temperature Run-to-Failure Dataset** by Jung et al. (2024) from Mendeley Data:
- [Dataset Link (Mendeley Data)](https://doi.org/10.17632/5hcdd3tdvb.6)

Extract the hourly CSV files into `data/raw/`:
```
WCA-GRU-Bearing-RUL/
└── data/
    └── raw/
        ├── LogFile_2022-06-20-17-00-31.csv
        ├── LogFile_2022-06-20-18-00-31.csv
        └── ... (129 files in total)
```

---

## 🚀 Usage

### 1. Data Preprocessing
Run the feature extraction and temporal splitting pipeline:
```bash
python src/preprocess.py
```
*Note: To change split mode (S1, S2, or S3), set `SPLIT_MODE` in `src/config.py`.*

### 2. Model Training & Evaluation
Train the WCA-GRU model and evaluate performance:
```bash
python src/train.py
```

---

## 📊 Performance Summary

| Model Variant | S1 Split (RMSE ↓ / MAE ↓) | S2 Split (RMSE ↓ / MAE ↓) | S3 PHM Score ↓ |
|---|:---:|:---:|:---:|
| **Case 1: 1D-CNN-GRU (Vib only)** | 42.15 / 38.60 | 28.60 / 26.35 | 24.78 |
| **Case 2: MLP-GRU (Temp only)** | 35.40 / 32.15 | 24.10 / 22.05 | 9.40 |
| **Case 3: Concat-GRU (Vib + Temp)** | 31.80 / 28.95 | 21.45 / 19.80 | 7.85 |
| **Case 4: WCA-GRU (Ours)** | **26.50 / 24.12** | **16.75 / 15.40** | **5.52** |

---

## 📜 Citation & Acknowledgements

If you use this repository or paper in your research, please cite:

```bibtex
@article{phan2026wcagru,
  title={A Multimodal Framework for Bearing Remaining Useful Life Prediction using Wavelet Cross-Attention GRU},
  author={Phan, Anh Minh and Nguyen, Xuan Trung and Le, Phu-Nguyen},
  journal={Predictive Maintenance & Machine Learning},
  year={2026}
}
```

*Funding:* This research is funded by Nguyen Tat Thanh University under grant number 2026.01.259.
