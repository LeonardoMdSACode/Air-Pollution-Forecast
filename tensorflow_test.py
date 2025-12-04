import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# CONFIG (tweak these if needed)
# ----------------------------
TRAIN_PATH = "LSTM-Multivariate_pollution.csv"
TEST_PATH  = "pollution_test_data1.csv"

TIME_STEPS = 12        # shorter lookback -> faster per epoch (try 12 or 8)
BATCH_SIZE = 128       # larger batches improve throughput
EPOCHS     = 50        # early stopping will usually stop earlier
PATIENCE   = 10
LR         = 1e-3

# ----------------------------
# LOAD & ALIGN COLUMNS
# ----------------------------
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

print("Train columns:", df_train.columns.tolist())
print("Test columns:", df_test.columns.tolist())

# Drop date column in train (test does not have it)
if "date" in df_train.columns:
    df_train = df_train.drop(columns=["date"])

# Ensure consistent column order
cols = ["pollution", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]
df_train = df_train[cols]
df_test  = df_test[cols]

# ----------------------------
# SIMPLE MISSING VALUE HANDLING
# ----------------------------
# numeric interpolation + forward/backward for category
num_cols = ["pollution","dew","temp","press","wnd_spd","snow","rain"]
df_train[num_cols] = df_train[num_cols].interpolate(limit_direction="both", axis=0)
df_test[num_cols]  = df_test[num_cols].interpolate(limit_direction="both", axis=0)

df_train["wnd_dir"] = df_train["wnd_dir"].ffill().bfill()
df_test["wnd_dir"] = df_test["wnd_dir"].ffill().bfill()

# ----------------------------
# PREPROCESSING (OneHot for wnd_dir + scaling)
# ----------------------------
# We'll build a ColumnTransformer: OneHot for wnd_dir, MinMax for numeric
numeric_features = ["pollution","dew","temp","press","wnd_spd","snow","rain"]
cat_features = ["wnd_dir"]

# OneHotEncoder (sparse -> False to get numpy array)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Fit OHE on combined categories to avoid unseen categories problems
ohe.fit(pd.concat([df_train[cat_features], df_test[cat_features]], axis=0))

# Build scikit-like transform manually: encode cat then scale numerics
def preprocess_df(df, fit_num_scaler=None):
    # encode categorical
    cat_enc = ohe.transform(df[cat_features])
    # numeric array
    num_arr = df[numeric_features].values.astype(np.float32)
    # scale numerics with MinMaxScaler: fit on train only
    if fit_num_scaler is None:
        scaler = MinMaxScaler()
        num_scaled = scaler.fit_transform(num_arr)
        return np.hstack([num_scaled, cat_enc]), scaler
    else:
        num_scaled = fit_num_scaler.transform(num_arr)
        return np.hstack([num_scaled, cat_enc])

# Fit scaler on train numeric features
from sklearn.preprocessing import MinMaxScaler
num_scaler = MinMaxScaler()
num_scaler.fit(df_train[numeric_features].values.astype(np.float32))

X_train_full = preprocess_df(df_train, fit_num_scaler=num_scaler)
X_test_full  = preprocess_df(df_test,  fit_num_scaler=num_scaler)

print("Processed feature dims (train/test):", X_train_full.shape, X_test_full.shape)

# We'll use the processed arrays directly. Note: pollution is present as first numeric column
# This means past pollution is available to the model as an input feature (strongly helpful).

# ----------------------------
# CREATE SEQUENCES (sliding window)
# ----------------------------
def create_sequences_from_array(X, time_steps):
    """
    X: 2D array (timesteps, features)
    returns: X_seq (samples, time_steps, features), y_next (samples,)
    target is the pollution value at t+time_steps (we assume pollution is index 0 in numeric block)
    """
    xs, ys = [], []
    for i in range(len(X) - time_steps):
        xs.append(X[i:i+time_steps])
        # target is the pollution (original scale is encoded in scaled features, but because we scaled all numeric features together,
        # we will inverse-target later using the numeric scaler.)
        # The pollution in scaled space is column 0 of the numeric block. Our processed X layout is:
        # [scaled_numeric_columns..., onehot_cat...]. So pollution scaled value is column index 0.
        ys.append(X[i+time_steps, 0])  # scaled pollution at next timestep
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xtr_seq, ytr_seq = create_sequences_from_array(X_train_full, TIME_STEPS)
Xte_seq, yte_seq = create_sequences_from_array(X_test_full, TIME_STEPS)

print("Xtr_seq shape:", Xtr_seq.shape, "ytr_seq shape:", ytr_seq.shape)
print("Xte_seq shape:", Xte_seq.shape, "yte_seq shape:", yte_seq.shape)

# ----------------------------
# tf.data pipeline for speed
# ----------------------------
def make_tf_dataset(X, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_tf_dataset(Xtr_seq, ytr_seq, BATCH_SIZE, shuffle=True)
val_split = 0.1
# create validation from last 10% of training sequences (simple split)
val_size = int(len(Xtr_seq) * val_split)
if val_size > 0:
    # we'll use last val_size samples for validation
    X_val = Xtr_seq[-val_size:]
    y_val = ytr_seq[-val_size:]
    X_tr = Xtr_seq[:-val_size]
    y_tr = ytr_seq[:-val_size]
    train_ds = make_tf_dataset(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(X_val, y_val, BATCH_SIZE, shuffle=False)
else:
    val_ds = None

# ----------------------------
# MODEL: Conv1D -> GRU -> Dense
# - Conv1D extracts local patterns (fast)
# - GRU is faster than LSTM
# ----------------------------
n_features = Xtr_seq.shape[2]

tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape=(TIME_STEPS, n_features))
x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.GRU(64, return_sequences=False)(x)  # faster than LSTM
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=tf.keras.losses.Huber(), metrics=["mae"])
model.summary()

# ----------------------------
# CALLBACKS
# ----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
]

# ----------------------------
# TRAIN
# ----------------------------
if val_ds is not None:
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
else:
    history = model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

# ----------------------------
# PREDICT ON TEST
# ----------------------------
y_pred_scaled = model.predict(Xte_seq, batch_size=BATCH_SIZE).flatten()
y_true_scaled = yte_seq.flatten()

# We scaled numeric features using num_scaler on full numeric array.
# The scaled pollution value corresponds to numeric_features index 0 scaled by num_scaler.
# To inverse scale the predictions (and true values), we need pollution's original scaling:
pollution_min = num_scaler.data_min_[0]
pollution_max = num_scaler.data_max_[0]
pollution_range = pollution_max - pollution_min
# For MinMaxScaler: X_scaled = (X - data_min) / (data_max - data_min)
# Inverse: X = X_scaled * (data_max - data_min) + data_min
y_pred = y_pred_scaled * pollution_range + pollution_min
y_true = y_true_scaled * pollution_range + pollution_min

# ----------------------------
# METRICS
# ----------------------------
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
try:
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
except:
    r2 = float("nan")

print("\nEVALUATION on TEST SEQUENCES")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# ----------------------------
# PLOTS: Predictions vs Actual (first N points)
# ----------------------------
N = min(300, len(y_true))
plt.figure(figsize=(12,4))
plt.plot(y_true[:N], label="Actual", linewidth=1)
plt.plot(y_pred[:N], label="Predicted", linewidth=1)
plt.title("PM2.5: Actual vs Predicted (first {} samples)".format(N))
plt.xlabel("Sample index")
plt.ylabel("PM2.5")
plt.legend()
plt.tight_layout()
plt.show()

# Training loss plot
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="train_loss")
if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (Huber)")
plt.legend()
plt.tight_layout()
plt.show()
