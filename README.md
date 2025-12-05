# Multivariate PM2.5 Forecasting with Conv1D + LSTM (TensorFlow)
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
## About Pollution_Multivariate_PyTorch.ipynb

---

# Multivariate PM2.5 Forecasting with Conv1D + LSTM (PyTorch Version)

---

## About Pollution_Multivariate_PyTorch.py

---

## Overview

This project provides a **PyTorch-based reimplementation** of the multivariate time series forecasting pipeline used to predict PM2.5 pollution levels from meteorological variables.
The architecture mirrors the TensorFlow version as closely as possible, using the same **Conv1D → LSTM hybrid model**, preprocessing pipeline, sequence generation logic, and evaluation metrics.

This PyTorch pipeline includes:

* Leak-free preprocessing using scikit-learn transformers
* Sliding-window sequence creation for supervised time series modeling
* Efficient PyTorch `DataLoader` batching
* A Conv1D + LSTM architecture faithful to the original TensorFlow model
* Huber loss and MAE monitoring during training
* Early stopping with **in-memory best weight tracking** (replicating TF behavior without saving model files)
* Post-training evaluation with MAE, MSE, RMSE, and R²

---

## Data

* **Training data:** `LSTM-Multivariate_pollution.csv`
* **Test data:** `pollution_test_data1.csv`

**Features used:**

* Numeric: `pollution`, `dew`, `temp`, `press`, `wnd_spd`, `snow`, `rain`
* Categorical: `wnd_dir`

Missing values are handled identically to the TensorFlow pipeline:

* Numeric features interpolated (bidirectional)
* Categorical feature (`wnd_dir`) forward/backward filled
* `date` column dropped if present

---

## Preprocessing

Preprocessing is performed using **scikit-learn**, exactly matching the TensorFlow pipeline:

* **MinMaxScaler** for numeric features (fit **only on training** to avoid leakage)
* **OneHotEncoder** for categorical feature (`handle_unknown="ignore"` to support unseen categories)
* A unified **ColumnTransformer** merges scaled numeric features and OHE categorical features
* Train and test data use the **same fitted transformer**, ensuring consistent feature ordering

---

## Sequence Creation

Time series sequences are generated using a sliding window:

* `TIME_STEPS = 12` past timesteps used as input
* The target is the **pollution value at the next timestep**
* Final training samples have shape:

  ```
  (num_samples, TIME_STEPS, n_features)
  ```

---

## Model Architecture (PyTorch)

This PyTorch model recreates the TensorFlow architecture:

* **Conv1D**:

  * 64 filters
  * kernel size 3
  * causal behavior simulated by padding + slicing
  * ReLU activation
* **BatchNorm + Dropout (0.2)**
* **LSTM (64 units)** returning last hidden state
* **BatchNorm + Dropout (0.2)**
* **Dense (FC) layer** with 32 units + ReLU
* **Output layer:** 1 unit (predict scaled PM2.5)

**Optimizer:** Adam (LR = 1e-3)
**Loss:** SmoothL1Loss (PyTorch’s Huber loss equivalent)
**Metric:** MAE

Early stopping is implemented manually with **in-memory best weight restoration**, mimicking Keras without saving files.

---

## Training & Validation

* Train/validation split: last 10% of sequences used as validation
* Batch size: 128
* Epochs: 50
* Early stopping patience: 10

The model restores the best-performing weights (lowest validation loss) after training ends.

---

## Prediction & Inverse Scaling

Because predictions are made on **scaled** data, pollution values must be inverse-transformed:

* Retrieve `data_min_` and `data_max_` from the numeric MinMaxScaler
* Convert predictions back to µg/m³
* Ensure comparability with real-world PM2.5 values

This yields true (unscaled) curves for visualization and metric computation.

---

## Evaluation on Test Data

| Metric | Value  |
| ------ | ------ |
| MAE    | 19.17  |
| MSE    | 870.87 |
| RMSE   | 29.51  |
| R²     | 0.911  |

Compared to the TensorFlow model:

* The PyTorch version performs **slightly worse**, with larger errors and lower R²
* This is expected due to subtle framework differences in:

  * initialization
  * BatchNorm behavior
  * Conv1D causal simulation
  * dropout timing
  * sequence handling

Even so, the model maintains strong performance and captures pollution trends well.

---

## Visualization

The script plots:

* **Actual vs Predicted PM2.5** for the first 300 test samples
* **Training and validation loss curves** to visualize convergence and early stopping

---

## Notes

* Preprocessing is entirely leak-free (train-only fitting)
* SmoothL1Loss provides robustness against spikes in pollution levels
* Best weights are restored using **in-memory state_dict() tracking**, not disk files

---

## License

MIT License
