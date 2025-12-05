# Multivariate PM2.5 Forecasting with Conv1D + LSTM
---
## About Pollution_Multivariate_TensorFlow.ipynb
---
## Overview

This project demonstrates a **multivariate time series forecasting pipeline** for predicting PM2.5 pollution using weather data. The model uses a combination of **1D Convolution + LSTM layers** implemented in TensorFlow/Keras.

The pipeline includes:

* Leak-free preprocessing (numeric scaling + categorical one-hot encoding)
* Sequence creation (sliding window) for time series modeling
* Efficient `tf.data` pipelines for batching and shuffling
* A deep learning model with Conv1D and LSTM layers
* Robust loss function (Huber) and monitoring with MAE
* Post-training evaluation including MAE, MSE, RMSE, and R²

---

## Data

* **Training data:** `LSTM-Multivariate_pollution.csv`
* **Test data:** `pollution_test_data1.csv`

**Features used:**

* Numeric: `pollution`, `dew`, `temp`, `press`, `wnd_spd`, `snow`, `rain`
* Categorical: `wnd_dir`

The `date` column is dropped, and numeric missing values are handled with interpolation, while categorical missing values are forward/backward filled.

---

## Preprocessing

* **Numeric features** are scaled using `MinMaxScaler` (fit on training data only).
* **Categorical features** are one-hot encoded using `OneHotEncoder` (fit on training data only, unseen categories in test set ignored).
* Preprocessing uses a **ColumnTransformer**, ensuring a leak-free transformation pipeline.
* Train and test sets are transformed using the same fitted preprocessor.

---

## Sequence Creation

* Time series sequences are created using a **sliding window** approach.
* `TIME_STEPS` (lookback) is set to 12.
* Each sequence has shape `(TIME_STEPS, n_features)` with target being **next timestep's pollution value**.

---

## Model Architecture

* **Input:** `(TIME_STEPS, n_features)`
* **Conv1D Layer:** 64 filters, kernel size 3, causal padding, ReLU activation
* **BatchNormalization + Dropout (0.2)**
* **LSTM Layer:** 64 units, returns final state
* **BatchNormalization + Dropout (0.2)**
* **Dense Layer:** 32 units, ReLU
* **Output Layer:** 1 unit (predict scaled pollution)

**Optimizer:** Adam with learning rate `1e-3`
**Loss:** Huber (robust to outliers)
**Metric:** MAE

Early stopping and learning rate reduction callbacks are used for stable training.

---

## Training & Validation

* Train/validation split: last 10% of training sequences used as validation.
* Batch size: 128
* Epochs: 50 (with early stopping)

---

## Prediction & Inverse Scaling

* Predictions and true values are **scaled** during preprocessing.
* After model prediction, pollution values are **inverse-scaled** using the training MinMaxScaler to return to original units (µg/m³).

---

## Evaluation on Test Data

| Metric | Value  |
| ------ | ------ |
| MAE    | 15.32  |
| MSE    | 748.53 |
| RMSE   | 27.36  |
| R²     | 0.923  |

* **MAE** (Mean Absolute Error) indicates an average prediction error of ~15.3 µg/m³.
* **R²** of 0.923 shows the model explains ~92% of the variance in PM2.5 levels.
* Overall, the LSTM-based model outperforms the previous GRU version and shows improved accuracy.

---

## Visualization

* Actual vs Predicted PM2.5 values for the first 300 test samples.
* Training and validation loss curves to monitor convergence and early stopping behavior.

---

## Notes

* The pipeline is **leak-free**: no information from the test set is used during preprocessing.
* The Huber loss is used to reduce the impact of outliers in pollution readings.
* The ColumnTransformer ensures consistent feature ordering for both numeric and categorical features.

---

## Usage

1. Place `LSTM-Multivariate_pollution.csv` and `pollution_test_data1.csv` in the project folder.
2. Install dependencies: `tensorflow`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`
3. Run the script: `python pm25_forecasting_lstm.py`
4. View printed evaluation metrics and plots.

---

## License

MIT License
