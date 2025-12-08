# Multivariate pollution Forecasting — Deep Learning (Conv1D + LSTM) and Classical Machine Learning

This repository provides a complete multivariate time series forecasting pipeline for predicting **air pollution** using meteorological data. Three complementary approaches are implemented:

1. **TensorFlow Conv1D + LSTM Model**
2. **PyTorch Conv1D + LSTM Model**
3. **Classic Machine Learning Regression Models (XGBoost, LightGBM, Random Forest, etc.)**

Each approach uses leak-free preprocessing, consistent feature engineering, separate training/testing, robust evaluation metrics, and visualizations.

---

# Data

Two datasets are used:

* **Training:** `LSTM-Multivariate_pollution.csv`
* **Testing:** `pollution_test_data1.csv`

### Features

**Numeric:** `pollution`, `dew`, `temp`, `press`, `wnd_spd`, `snow`, `rain`
**Categorical:** `wnd_dir`

Notes:

* The `date` column is dropped if present.
* Numeric missing values are interpolated.
* Categorical missing values are forward/backward filled.

---

# 1. TensorFlow Model — Conv1D + LSTM

Files:

* `Pollution_Multivariate_TensorFlow.ipynb`
* `pm25_forecasting_lstm.py`

### Model Architecture

```
Input → Conv1D → BatchNorm → Dropout
      → LSTM → BatchNorm → Dropout
      → Dense(32) → Dense(1)
```

**Key settings:** Conv1D(64), LSTM(64), Dropout(0.2), Huber loss, Adam(1e-3), 12 time steps.

### Evaluation

* MAE: 15.32
* MSE: 748.53
* RMSE: 27.36
* R²: 0.923

---

# 2. PyTorch Model — Conv1D + LSTM

Files:

* `Pollution_Multivariate_PyTorch.ipynb`
* `Pollution_Multivariate_PyTorch.py`

### Architecture

```
Input → Conv1D → BatchNorm → Dropout
      → LSTM → BatchNorm → Dropout
      → Dense(32) → Dense(1)
```

### Evaluation

* MAE: 19.17
* MSE: 870.87
* RMSE: 29.51
* R²: 0.911

---

# 3. Classic Machine Learning Models

File:

* `Pollution_Multivariate_Classic_Models.py`

### Feature Engineering

For each numeric column:

* Lag features: `lag1` → `lag12`
* Rolling stats: mean, std, min, max (window=3)

### Models

Linear Regression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, GradientBoosting, **XGBoost**, **LightGBM**.

### Evaluation (Top Models)

| Model        | MAE      | RMSE      | R²         |
| ------------ | -------- | --------- | ---------- |
| **XGBoost**  | **8.16** | **12.52** | **0.9839** |
| LightGBM     | 11.55    | 20.54     | 0.9568     |
| RandomForest | 12.66    | 21.95     | 0.9506     |

XGBoost outperforms all deep learning and classic models.

---

# Cross-Model Comparison

| Model Type               | Best MAE | Best RMSE | Best R²   |
| ------------------------ | -------- | --------- | --------- |
| **Classic ML (XGBoost)** | **8.16** | **12.52** | **0.984** |
| TensorFlow Conv1D+LSTM   | 15.32    | 27.36     | 0.923     |
| PyTorch Conv1D+LSTM      | 19.17    | 29.51     | 0.911     |

---

# Usage

```
pip install tensorflow torch scikit-learn xgboost lightgbm numpy pandas matplotlib
```

Run:

```
python Pollution_Multivariate_TensorFlow.py
python Pollution_Multivariate_PyTorch.py
python Pollution_Multivariate_Classic_Models.py
```

---

# License

MIT License
