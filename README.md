# Wavelet Cross-Attention GRU (WCA-GRU) for Bearing RUL Prediction

Research implementation associated with the manuscript:
**"A Multimodal Framework for Bearing Remaining Useful Life Prediction using Wavelet Cross-Attention GRU"**
*Authors:* Phan Anh Minh, Nguyen Xuan Trung, Phu-Nguyen Le.

[![CI](https://github.com/xuantrung-DA/bearing-rul-multimodal-eidt-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/xuantrung-DA/bearing-rul-multimodal-eidt-2026/actions/workflows/ci.yml)

**Evidence status:** this source release does not distribute the historical
split manifests, predictions, or run/checkpoint hashes needed to audit the former
performance table. That table is no longer presented as a verified benchmark.
The corrected concatenation and single-modality baselines require new training;
no replacement metrics are claimed in this maintenance update.

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
git clone https://github.com/xuantrung-DA/bearing-rul-multimodal-eidt-2026.git
cd bearing-rul-multimodal-eidt-2026
```

### 2. Install dependencies
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
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

No audited performance summary is distributed with this source release.
Use `MODALITY_MODE` in [src/config.py](src/config.py) to construct a new model:

| Mode | Architecture contract |
| --- | --- |
| `vib_only` | Vibration encoder -> GRU -> regression head; no thermal encoder or fusion |
| `temp_only` | Thermal MLP -> GRU -> regression head; no vibration encoder or fusion |
| `concat` | Concatenate both encoder outputs along features -> GRU -> head |
| `wca_gru` | Vibration query, thermal key/value -> cross-attention -> GRU -> head |

The former `concat` path added embeddings rather than concatenating them, and
single-modality modes zeroed inputs while retaining the fusion path. Old baseline
checkpoints/results are not compatible with the corrected comparisons. The main
`wca_gru` model architecture is unchanged. Do not change modality after constructing
a model; instantiate a fresh model and use a separate output directory per mode.

Synthetic tests verify architecture contracts, not RUL accuracy. A new benchmark
must publish file-level splits, scaler fit scope, seeds, dependency versions,
parameter counts, per-origin/per-run metrics, predictions and checkpoint hashes.
Default file-level temporal splits do not establish generalization to unseen
bearings. Requirements specify minimum versions, not a full dependency lock.

---

## 📜 Citation & Acknowledgements

Until a final publisher record is available and verified, cite the manuscript
without inventing a journal, DOI, or publication status:

```bibtex
@misc{phan2026wcagru,
  title={A Multimodal Framework for Bearing Remaining Useful Life Prediction using Wavelet Cross-Attention GRU},
  author={Phan, Anh Minh and Nguyen, Xuan Trung and Le, Phu-Nguyen},
  note={Research manuscript; final publication metadata not verified in this repository},
  year={2026}
}
```

*Funding:* This research is funded by Nguyen Tat Thanh University under grant number 2026.01.259.
