# Predictive Maintenance – Turbine Failure Prediction

End-to-End Machine Learning project: From sensor data to production-ready REST API.

---

## What does this project do?

This system predicts whether a turbine will fail within the next 30 operating cycles based on sensor data.

**Input:** 50 timesteps × 24 sensors (temperature, pressure, rotation speed, etc.)

**Output:** Risk score (0–100%), risk level (low/medium/high), alert flag

**Use case:** Early maintenance planning before failure occurs.

---

## Results

| Metric | Validation | Test |
|--------|------------|------|
| PR-AUC | 0.974 | 0.984 |
| Precision | 0.932 | 0.966 |
| Recall | 0.851 | 0.882 |
| F1 | 0.890 | 0.922 |

Model: LightGBM (selected over Logistic Regression by PR-AUC on validation)

Threshold: 0.8974 (selected to satisfy recall ≥ 85% and precision ≥ 30%)

---

## Architecture
```
Data (NASA C-MAPSS) → Labeling → Engine-Split (70/15/15) → Training → Threshold Selection → REST API
```

---

## Features

**Data Pipeline**
- Raw data parsing and preprocessing
- Binary labeling: failure within horizon yes/no
- Engine-based train/val/test split to prevent leakage

**Feature Engineering**
- Tabular: rolling statistics, lags, deltas
- Sequence: sliding windows for CNN/GRU input
- Scaler fit on training data only

**Models**
- Logistic Regression (baseline)
- LightGBM (best performing)
- 1D-CNN (PyTorch)
- GRU (PyTorch)
- IsolationForest (anomaly detection baseline)

**Evaluation**
- PR-AUC as primary metric (handles class imbalance)
- Configurable threshold policy (target recall + minimum precision)
- Calibration analysis and backtesting

**Analysis & Reports**
- EDA notebook with sensor correlation, degradation plots, class imbalance analysis
- Evaluation reports for baseline and anomaly models

**Serving**
- FastAPI REST API with Pydantic validation
- Returns risk score, risk bucket, and alert flag
- Input validation (exact window length, no NaN/Inf)

**MLOps**
- MLflow experiment tracking
- AWS S3 artifact storage
- Docker and docker-compose deployment
- Inference benchmark script

**Testing**
- 24 unit test files
- Coverage: split logic, feature leakage, API contracts, threshold policy

---

## Quickstart
```bash
git clone https://github.com/danielsamo-se/pm-risk.git
cd pm-risk

python -m venv .venv
source .venv/bin/activate
pip install -e ".[serving,dl,data]"

python -m pmrisk.data.make_processed
python -m pmrisk.labeling.labels
python -m pmrisk.models.train_tabular

uvicorn pmrisk.serving.app:app --reload
```

---

## Docker
```bash
docker-compose up --build
```

API runs on http://localhost:8000

MinIO console on http://localhost:9001

---

## API

### GET /health

Returns API status, model version, and model type.

### POST /predict

**Request:** JSON with "window" containing exactly 50 rows of sensor data.

**Response:**

| Field | Description |
|-------|-------------|
| risk_score | Failure probability (0.0 – 1.0) |
| bucket | Risk level: low (<0.2), med (0.2–0.5), high (>0.5) |
| threshold | Configured threshold value |
| is_alert | True if risk_score ≥ threshold |
| model_version | Active model version |
| model_type | Model type (e.g. lightgbm, gru) |

Returns 422 for invalid input.

---

## Configuration

All parameters configurable via YAML files:

| Parameter | File | Value | Description |
|-----------|------|-------|-------------|
| horizon_n | base.yaml | 30 | Prediction horizon (cycles) |
| window_l | base.yaml | 50 | Input window length |
| target_recall | thresholds.yaml | 0.85 | Minimum recall for threshold |
| min_precision | thresholds.yaml | 0.30 | Minimum precision for threshold |
| bucket_cutoffs | thresholds.yaml | [0.2, 0.5] | Risk level boundaries |
| train_ratio | split.yaml | 0.70 | Training set ratio |

---

## Design Decisions

**Engine-based Split:** Each engine is assigned to exactly one split to prevent data leakage. No engine appears in multiple splits.

**PR-AUC Metric:** Dataset is imbalanced (~19% positive class). PR-AUC correctly evaluates performance on the minority class.

**Threshold Policy:** Threshold satisfies recall ≥ 85% AND precision ≥ 30%. Values are configurable for different operational requirements.

**Model Selection:** Both Logistic Regression and LightGBM are trained. The model with higher PR-AUC on validation is selected automatically.

---

## Dataset

NASA C-MAPSS FD001: Turbofan engine degradation simulation with 100 engines, ~20,000 cycles, and 21 sensors.

---

## Tech Stack

- ML: PyTorch, LightGBM, scikit-learn
- Data: Pandas, NumPy, PyArrow
- API: FastAPI, Pydantic, Uvicorn
- MLOps: MLflow, Docker, AWS S3
- Testing: pytest

---

## License

MIT