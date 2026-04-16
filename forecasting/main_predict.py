import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import redis

r = redis.Redis(host='192.168.88.168', port=6379, db=0)

def get_data_from_redis():
    ts = r.ts()

    # Build one DataFrame per sensor key: [timestamp_ms, value]
    frames = []
    keys = sorted(ts.execute_command("KEYS", "sensor:264041591600404:*"))

    for key in keys:
        key_str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        data = ts.range(key, from_time='-', to_time='+', bucket_size_msec=5 * 60 * 1000)
        if not data:
            continue

        key_df = pd.DataFrame(data, columns=["timestamp_ms", key_str])
        frames.append(key_df)

    if not frames:
        return pd.DataFrame(columns=["timestamp_ms"])

    # Outer merge on timestamp so sensors with missing points are preserved
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="timestamp_ms", how="outer")

    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df




CSV_PATH = "data.csv"

LOOKBACK = 72        # number of past steps used as input
HORIZON = 12         # number of future steps to predict
RAIN_THRESHOLD = 0.1    # mm threshold to consider it as rain in evaluation
TRAIN_RATIO = 0.8
VAL_RATIO = 0.15
TARGET_COL = "sensor:264041591600404:temperature"

FEATURE_COLS = [
    "sensor:264041591600404:temperature",
    "sensor:264041591600404:humidity",
    "sensor:264041591600404:pressure",
    "sensor:264041591600404:wind_speed",
    "sensor:264041591600404:rainfall",
    "sensor:264041591600404:wind_direction_degrees",
]




def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Fallback: if index got saved accidentally as unnamed column
    if "timestamp_ms" not in df.columns and "Unnamed: 0" in df.columns:
        maybe_ts = pd.to_numeric(df["Unnamed: 0"], errors="coerce")
        if maybe_ts.notna().all():
            df["timestamp_ms"] = maybe_ts.astype("int64")

    if "timestamp_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("CSV must contain either 'timestamp_ms' or 'timestamp'.")

    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[["timestamp"] + FEATURE_COLS].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Force numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic interpolation / cleanup
    df = df.interpolate(method="linear").ffill().bfill()

    # Encode wind direction as sin/cos
    wind_rad = np.deg2rad(df["sensor:264041591600404:wind_direction_degrees"].values)
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)

    # Final model features
    model_features = [
        "sensor:264041591600404:temperature",
        "sensor:264041591600404:humidity",
        "sensor:264041591600404:pressure",
        "sensor:264041591600404:wind_speed",
        "sensor:264041591600404:rainfall",
        "wind_dir_sin",
        "wind_dir_cos",
    ]

    return df, model_features


def split_and_scale(df, model_features, train_ratio=0.8):
    values = df[model_features].values.astype(np.float32)

    train_size = int(len(values) * train_ratio)
    train_values = values[:train_size]
    test_values = values[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    return train_scaled, test_scaled, scaler


def create_sequences(data_scaled, lookback, horizon, target_idx=0):
    X, y = [], []

    for i in range(lookback, len(data_scaled) - horizon + 1):
        X.append(data_scaled[i - lookback:i, :])
        y.append(data_scaled[i:i + horizon, target_idx])

    return np.array(X), np.array(y)


def build_model(n_steps, n_features, horizon):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(horizon)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


def inverse_transform_temperature_only(y_scaled, scaler, n_features, target_idx=0):
    temp = np.zeros((len(y_scaled), n_features), dtype=np.float32)
    temp[:, target_idx] = y_scaled
    inv = scaler.inverse_transform(temp)
    return inv[:, target_idx]


def evaluate_forecasts(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def forecast_next_horizon(model, recent_scaled_window, scaler, n_features):
    pred_scaled = model.predict(recent_scaled_window[np.newaxis, :, :], verbose=0)[0]

    temp = np.zeros((len(pred_scaled), n_features), dtype=np.float32)
    temp[:, 0] = pred_scaled
    pred_real = scaler.inverse_transform(temp)[:, 0]

    return pred_real


if __name__ == "__main__":
    df = get_data_from_redis()
    print(df.head())

    # Keep timestamp_ms as a normal column in CSV
    df.to_csv("data.csv", index=False)

    df, model_features = load_and_prepare_data('data.csv')
    train_scaled, test_scaled, scaler = split_and_scale(df, model_features, train_ratio=TRAIN_RATIO)

    n_features = len(model_features)

    X_train, y_train = create_sequences(train_scaled, LOOKBACK, HORIZON, target_idx=0)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK, HORIZON, target_idx=0)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    model = build_model(LOOKBACK, n_features, HORIZON)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate first-step forecast quality on test set
    y_pred = model.predict(X_test, verbose=0)

    y_test_step1 = inverse_transform_temperature_only(y_test[:, 0], scaler, n_features)
    y_pred_step1 = inverse_transform_temperature_only(y_pred[:, 0], scaler, n_features)

    rmse, mae = evaluate_forecasts(y_test_step1, y_pred_step1)
    print(f"Test RMSE (step+1 temperature): {rmse:.4f}")
    print(f"Test MAE  (step+1 temperature): {mae:.4f}")

    # Plot first-step predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_step1[:300], label="Actual")
    plt.plot(y_pred_step1[:300], label="Predicted")
    plt.title("Temperature Forecast - First Future Step")
    plt.xlabel("Sample")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pred_vs_actual.png")

    # Forecast next horizon from most recent LOOKBACK window
    full_values = df[model_features].values.astype(np.float32)
    full_scaled = scaler.transform(full_values)
    recent_window = full_scaled[-LOOKBACK:, :]

    future_temp = forecast_next_horizon(model, recent_window, scaler, n_features)

    print("\nNext forecasted temperature values:")
    for i, val in enumerate(future_temp, start=1):
        print(f"t+{i}: {val:.3f}")