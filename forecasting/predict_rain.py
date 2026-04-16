import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import redis
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
REDIS_HOST = "192.168.88.168"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PATTERN = "sensor:264041591600404:*"

BUCKET_SIZE_MSEC = 5 * 60 * 1000

LOOKBACK = 48              # 48 x 5 min = 4 hours, easier baseline
TEMP_HORIZON = 1           # start with 5-minute ahead temp forecast
RAIN_HORIZON = 12          # 1 hour rain event horizon
RAIN_THRESHOLD = 0.1
FORECAST_WINDOW = 60       # 60 x 5 min = 5 hours

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

MIN_RAIN_POSITIVES_TRAIN = 20
MIN_RAIN_POSITIVES_VAL = 5

OUTPUT_DIR = "debug_plots"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# -------------------------------------------------------------------
# REDIS LOADING
# -------------------------------------------------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def get_data_from_redis(pattern=REDIS_PATTERN, bucket_size_msec=BUCKET_SIZE_MSEC):
    ts = r.ts()
    raw_keys = ts.execute_command("KEYS", pattern)

    series_frames = []

    for key in raw_keys:
        key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)

        data = ts.range(
            key,
            from_time="-",
            to_time="+",
            aggregation_type="avg",
            bucket_size_msec=bucket_size_msec,
        )

        if not data:
            continue

        metric_name = key_str.split(":")[-1]
        timestamps, values = zip(*data)

        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, unit="ms"),
            metric_name: pd.to_numeric(values, errors="coerce"),
        })
        series_frames.append(frame)

    if not series_frames:
        return pd.DataFrame()

    df = series_frames[0]
    for frame in series_frames[1:]:
        df = df.merge(frame, on="timestamp", how="outer")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    required_cols = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rainfall",
        "wind_direction_degrees",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-only fill avoids leaking future information into earlier rows.
    df[required_cols] = df[required_cols].interpolate(method="linear", limit_direction="forward").ffill()
    df[required_cols] = df[required_cols].fillna(df[required_cols].median())

    wind_rad = np.deg2rad(df["wind_direction_degrees"])
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)

    df["temp_diff_1"] = df["temperature"].diff().fillna(0.0)
    df["humidity_diff_1"] = df["humidity"].diff().fillna(0.0)
    df["pressure_diff_1"] = df["pressure"].diff().fillna(0.0)
    df["rainfall_diff_1"] = df["rainfall"].diff().fillna(0.0)

    df["temp_roll_mean_3"] = df["temperature"].rolling(3, min_periods=1).mean()
    df["humidity_roll_mean_3"] = df["humidity"].rolling(3, min_periods=1).mean()
    df["pressure_roll_mean_3"] = df["pressure"].rolling(3, min_periods=1).mean()

    return df


def make_targets(df: pd.DataFrame, temp_horizon: int, rain_horizon: int, rain_threshold: float):
    df = df.copy()

    temp_target = []
    rain_target = []

    temp_values = df["temperature"].values
    rain_values = df["rainfall"].values

    for i in range(len(df)):
        if i + max(temp_horizon, rain_horizon) >= len(df):
            temp_target.append(np.nan)
            rain_target.append(np.nan)
            continue

        future_temp = temp_values[i + temp_horizon]
        future_rain_window = rain_values[i + 1:i + rain_horizon + 1]
        future_rain_max = np.max(future_rain_window)

        temp_target.append(float(future_temp))
        rain_target.append(1.0 if future_rain_max >= rain_threshold else 0.0)

    df["temp_target"] = temp_target
    df["rain_target"] = rain_target
    df = df.dropna().reset_index(drop=True)

    return df


# -------------------------------------------------------------------
# SEQUENCE BUILDING
# -------------------------------------------------------------------
def chronological_split_indices(n_rows, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return train_end, val_end


def fit_feature_scaler_on_train(df, feature_cols, train_end_row):
    scaler = StandardScaler()
    scaler.fit(df.loc[:train_end_row - 1, feature_cols])
    return scaler


def apply_feature_scaler(df, feature_cols, scaler):
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def fit_temp_target_scaler_on_train(df, train_end_row):
    scaler = StandardScaler()
    scaler.fit(df.loc[:train_end_row - 1, ["temp_target"]])
    return scaler


def apply_temp_target_scaler(df, scaler):
    df = df.copy()
    df["temp_target_scaled"] = scaler.transform(df[["temp_target"]]).astype(np.float32)
    return df


def build_temp_sequences(df, feature_cols, lookback):
    X, y, target_timestamps = [], [], []

    values = df[feature_cols].values
    y_values = df["temp_target_scaled"].values
    ts_values = df["timestamp"].values

    for i in range(lookback, len(df)):
        X.append(values[i - lookback:i])
        y.append(y_values[i])
        target_timestamps.append(ts_values[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(target_timestamps)


def build_rain_sequences(df, feature_cols, lookback):
    X, y, target_timestamps = [], [], []

    values = df[feature_cols].values
    y_values = df["rain_target"].values
    ts_values = df["timestamp"].values

    for i in range(lookback, len(df)):
        X.append(values[i - lookback:i])
        y.append(y_values[i])
        target_timestamps.append(ts_values[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(target_timestamps)


def build_sequences_for_target_range(df, feature_cols, lookback, target_col, start_idx, end_idx):
    """
    Build sequences where target index i satisfies start_idx <= i < end_idx.
    This preserves strict chronological splits and prevents train/val target leakage.
    """
    X, y, ts = [], [], []
    feature_values = df[feature_cols].values
    y_values = df[target_col].values
    ts_values = df["timestamp"].values

    effective_start = max(start_idx, lookback)
    for i in range(effective_start, end_idx):
        X.append(feature_values[i - lookback:i])
        y.append(y_values[i])
        ts.append(ts_values[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(ts)


# -------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------
def build_temp_model(n_steps, n_features):
    inp = Input(shape=(n_steps, n_features))

    x = LSTM(
        16,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(inp)
    x = Dropout(0.35)(x)
    x = LSTM(
        8,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(x)
    x = Dropout(0.35)(x)

    x = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    out = Dense(1, name="temp_out")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_rain_model(n_steps, n_features):
    inp = Input(shape=(n_steps, n_features))

    x = LSTM(
        16,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(inp)
    x = Dropout(0.35)(x)
    x = LSTM(
        8,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(x)
    x = Dropout(0.35)(x)

    x = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    out = Dense(1, activation="sigmoid", name="rain_out")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------
def inverse_temp_predictions(y_scaled, scaler):
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return scaler.inverse_transform(y_scaled).flatten()


def evaluate_temperature(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def evaluate_rain(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1


# -------------------------------------------------------------------
# FORECAST CSV
# -------------------------------------------------------------------
def generate_forecast_csv(
    df_scaled,
    df_original,
    feature_cols,
    lookback,
    forecast_steps,
    temp_model,
    rain_model,
    calibrator,
    feature_scaler,
    output_path="forecast.csv",
):
    """
    Produce a CSV with one row per 5-min step into the future (up to
    forecast_steps).  Each row contains:
      - forecast_timestamp, minutes_ahead
      - predicted_temperature_C, rain_probability
      - vector_avg_wind_speed, vector_avg_wind_direction_deg

    Wind columns are computed via vector averaging over the historical
    data that falls inside each forecast row's lookback window.
    """
    last_ts = pd.Timestamp(df_original["timestamp"].iloc[-1])
    bucket_td = pd.Timedelta(minutes=5)

    # Start from the last LOOKBACK rows of the scaled features.
    window = df_scaled[feature_cols].values[-lookback:].copy()

    # We also need the unscaled last row for updating raw values.
    last_raw = df_original[feature_cols].iloc[-1].values.copy()

    # Pre-compute historical vector-averaged wind from the raw data
    # (last LOOKBACK rows) to attach to every forecast row.
    hist = df_original.tail(lookback).copy()
    ws = pd.to_numeric(hist["wind_speed"], errors="coerce").fillna(0.0).values
    wd = np.deg2rad(pd.to_numeric(hist["wind_direction_degrees"], errors="coerce").fillna(0.0).values)
    mean_u = np.mean(-ws * np.sin(wd))
    mean_v = np.mean(-ws * np.cos(wd))
    hist_wind_speed = round(float(np.sqrt(mean_u ** 2 + mean_v ** 2)), 2)
    hist_wind_dir = round(float(np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360), 1)

    rows = []
    for step in range(1, forecast_steps + 1):
        inp = window[np.newaxis, :, :]  # (1, LOOKBACK, n_features)

        # temperature prediction (in °C, unscaled target)
        pred_temp = float(temp_model.predict(inp, verbose=0)[0, 0])

        # rain probability
        if rain_model is not None:
            rain_raw = float(rain_model.predict(inp, verbose=0)[0, 0])
            rain_prob = float(calibrator.transform([rain_raw])[0]) if calibrator is not None else rain_raw
        else:
            rain_prob = np.nan

        forecast_ts = last_ts + bucket_td * step

        rows.append({
            "forecast_timestamp": forecast_ts,
            "minutes_ahead": step * 5,
            "predicted_temperature_C": round(pred_temp, 2),
            "rain_probability": round(rain_prob, 4) if not np.isnan(rain_prob) else np.nan,
            "vector_avg_wind_speed": hist_wind_speed,
            "vector_avg_wind_direction_deg": hist_wind_dir,
        })

        # Roll the window forward: update temperature in the raw vector,
        # re-scale, and shift the window by one step.
        temp_col_idx = feature_cols.index("temperature")
        new_raw = last_raw.copy()
        new_raw[temp_col_idx] = pred_temp
        new_scaled = feature_scaler.transform(
            pd.DataFrame([new_raw], columns=feature_cols)
        ).flatten()
        window = np.vstack([window[1:], new_scaled])

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    return str(output_path)


# -------------------------------------------------------------------
# WIND VECTOR AVERAGING CSV
# -------------------------------------------------------------------
def generate_wind_csv(
    df_original,
    period_hours=24,
    output_path="wind_average.csv",
):
    """
    Compute a single vector-averaged wind direction and speed over the
    last *period_hours* hours of data.  Produces one summary row.

    Vector averaging:
        u = -speed * sin(dir_rad)
        v = -speed * cos(dir_rad)
        avg_speed = sqrt(mean_u² + mean_v²)
        avg_dir   = atan2(-mean_u, -mean_v)  mapped to [0, 360)
    """
    df = df_original.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cutoff = df["timestamp"].max() - pd.Timedelta(hours=period_hours)
    df = df[df["timestamp"] >= cutoff]

    speed = pd.to_numeric(df["wind_speed"], errors="coerce").fillna(0.0).values
    direction_deg = pd.to_numeric(df["wind_direction_degrees"], errors="coerce").fillna(0.0).values
    direction_rad = np.deg2rad(direction_deg)

    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)

    mean_u = np.mean(u)
    mean_v = np.mean(v)

    avg_speed = round(float(np.sqrt(mean_u ** 2 + mean_v ** 2)), 2)
    avg_dir = round(float(np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360), 1)
    scalar_avg = round(float(np.mean(speed)), 2)

    out_df = pd.DataFrame([{
        "period_hours": period_hours,
        "from": df["timestamp"].min(),
        "to": df["timestamp"].max(),
        "vector_avg_speed": avg_speed,
        "vector_avg_direction_deg": avg_dir,
        "scalar_avg_speed": scalar_avg,
    }])
    out_df.to_csv(output_path, index=False)
    return str(output_path)


# -------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------
def ensure_output_dir(output_dir=OUTPUT_DIR):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_temp_training(history, output_dir=OUTPUT_DIR):
    path = ensure_output_dir(output_dir)
    fig = plt.figure(figsize=(11, 4.8))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Temperature Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = path / "temperature_training_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_temp_mae(history, output_dir=OUTPUT_DIR):
    path = ensure_output_dir(output_dir)
    fig = plt.figure(figsize=(11, 4.8))
    plt.plot(history.history["mae"], label="train_mae")
    plt.plot(history.history["val_mae"], label="val_mae")
    plt.title("Temperature MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (scaled target units)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = path / "temperature_mae.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_temp_forecast(y_true, y_pred, output_dir=OUTPUT_DIR):
    path = ensure_output_dir(output_dir)
    fig = plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="actual_temp", linewidth=2)
    plt.plot(y_pred, label="pred_temp", linewidth=2)
    plt.title("Temperature Forecast on Test Set")
    plt.xlabel("Test sample index")
    plt.ylabel("Temperature (deg C)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = path / "temperature_forecast_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_rain_diag(y_true, y_prob, threshold=0.5, output_dir=OUTPUT_DIR):
    path = ensure_output_dir(output_dir)
    y_pred = (y_prob >= threshold).astype(int)

    fig = plt.figure(figsize=(14, 5))
    plt.plot(y_prob, label="rain_prob_calibrated", color="tab:blue")
    plt.plot(y_true, label="rain_true", color="tab:green", alpha=0.8)
    plt.scatter(np.arange(len(y_pred)), y_pred, label="rain_pred_label", color="tab:red", s=16, alpha=0.7)
    plt.axhline(threshold, color="gray", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.2f}")
    plt.title("Rain Probability vs True Events")
    plt.xlabel("Test sample index")
    plt.ylabel("Probability / Binary Label")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = path / "rain_probability_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    df = get_data_from_redis()

    if df.empty:
        raise RuntimeError(f"No data loaded from Redis for pattern {REDIS_PATTERN}")

    if "timestamp" not in df.columns:
        raise KeyError("Missing timestamp column after Redis merge.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = prepare_features(df)
    df = make_targets(
        df,
        temp_horizon=TEMP_HORIZON,
        rain_horizon=RAIN_HORIZON,
        rain_threshold=RAIN_THRESHOLD,
    )

    print("Loaded rows after feature/target prep:", len(df))
    print("Rain target distribution:")
    print(df["rain_target"].value_counts(dropna=False))
    print("\nTemperature summary:")
    print(df["temperature"].describe())

    feature_cols = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rainfall",
        "wind_dir_sin",
        "wind_dir_cos",
        "temp_diff_1",
        "humidity_diff_1",
        "pressure_diff_1",
        "rainfall_diff_1",
        "temp_roll_mean_3",
        "humidity_roll_mean_3",
        "pressure_roll_mean_3",
    ]

    train_end_row, val_end_row = chronological_split_indices(len(df), TRAIN_RATIO, VAL_RATIO)

    feature_scaler = fit_feature_scaler_on_train(df, feature_cols, train_end_row)

    df_scaled = apply_feature_scaler(df, feature_cols, feature_scaler)

    # -------------------------------
    # TEMPERATURE MODEL
    # -------------------------------
    X_temp_train, y_temp_train, ts_temp_train = build_sequences_for_target_range(
        df_scaled,
        feature_cols,
        LOOKBACK,
        "temp_target",
        start_idx=0,
        end_idx=train_end_row,
    )
    X_temp_val, y_temp_val, ts_temp_val = build_sequences_for_target_range(
        df_scaled,
        feature_cols,
        LOOKBACK,
        "temp_target",
        start_idx=train_end_row,
        end_idx=val_end_row,
    )
    X_temp_test, y_temp_test, ts_temp_test = build_sequences_for_target_range(
        df_scaled,
        feature_cols,
        LOOKBACK,
        "temp_target",
        start_idx=val_end_row,
        end_idx=len(df_scaled),
    )

    print("\nTemperature sequence shapes:")
    print("train:", X_temp_train.shape, y_temp_train.shape)
    print("val:  ", X_temp_val.shape, y_temp_val.shape)
    print("test: ", X_temp_test.shape, y_temp_test.shape)

    temp_model = build_temp_model(X_temp_train.shape[1], X_temp_train.shape[2])

    temp_early_stop = EarlyStopping(
        monitor="val_mae",
        patience=8,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )

    temp_history = temp_model.fit(
        X_temp_train,
        y_temp_train,
        validation_data=(X_temp_val, y_temp_val),
        epochs=60,
        batch_size=32,
        callbacks=[temp_early_stop],
        shuffle=False,
        verbose=1,
    )

    y_temp_test_real = y_temp_test
    y_temp_pred_test_real = temp_model.predict(X_temp_test, verbose=0).flatten()

    temp_mae, temp_rmse = evaluate_temperature(y_temp_test_real, y_temp_pred_test_real)

    print("\nTemperature metrics:")
    print(f"MAE:  {temp_mae:.4f}")
    print(f"RMSE: {temp_rmse:.4f}")

    print("\nLatest temperature forecast:")
    latest_temp_window = X_temp_test[-1:].copy() if len(X_temp_test) > 0 else X_temp_val[-1:].copy()
    latest_temp_pred_real = float(temp_model.predict(latest_temp_window, verbose=0)[0, 0])
    print(f"t+{TEMP_HORIZON} predicted temperature: {latest_temp_pred_real:.3f} deg C")

    temp_loss_plot = plot_temp_training(temp_history)
    temp_mae_plot = plot_temp_mae(temp_history)
    temp_forecast_plot = plot_temp_forecast(y_temp_test_real, y_temp_pred_test_real)

    # -------------------------------
    # RAIN MODEL (OPTIONAL)
    # -------------------------------
    X_rain_train, y_rain_train, ts_rain_train = build_sequences_for_target_range(
        df_scaled,
        feature_cols,
        LOOKBACK,
        "rain_target",
        start_idx=0,
        end_idx=train_end_row,
    )
    X_rain_val, y_rain_val, ts_rain_val = build_sequences_for_target_range(
        df_scaled,
        feature_cols,
        LOOKBACK,
        "rain_target",
        start_idx=train_end_row,
        end_idx=val_end_row,
    )
    X_rain_test, y_rain_test, ts_rain_test = build_sequences_for_target_range(
        df_scaled,
        feature_cols,
        LOOKBACK,
        "rain_target",
        start_idx=val_end_row,
        end_idx=len(df_scaled),
    )

    rain_pos_train = int(np.sum(y_rain_train == 1.0))
    rain_pos_val = int(np.sum(y_rain_val == 1.0))
    rain_pos_test = int(np.sum(y_rain_test == 1.0))

    print("\nRain positive counts:")
    print(f"train positives: {rain_pos_train}")
    print(f"val positives:   {rain_pos_val}")
    print(f"test positives:  {rain_pos_test}")

    trained_rain_model = None
    trained_calibrator = None

    if rain_pos_train >= MIN_RAIN_POSITIVES_TRAIN and rain_pos_val >= MIN_RAIN_POSITIVES_VAL:
        rain_model = build_rain_model(X_rain_train.shape[1], X_rain_train.shape[2])

        classes = np.array([0.0, 1.0])
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_rain_train.astype(int),
        )
        class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}

        rain_early_stop = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        )

        rain_history = rain_model.fit(
            X_rain_train,
            y_rain_train,
            validation_data=(X_rain_val, y_rain_val),
            epochs=60,
            batch_size=32,
            callbacks=[rain_early_stop],
            class_weight=class_weight_dict,
            shuffle=False,
            verbose=1,
        )

        rain_prob_val = rain_model.predict(X_rain_val, verbose=0).flatten()
        rain_prob_test = rain_model.predict(X_rain_test, verbose=0).flatten()

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(rain_prob_val, y_rain_val)
        rain_prob_test_cal = calibrator.transform(rain_prob_test)

        rain_precision, rain_recall, rain_f1 = evaluate_rain(y_rain_test, rain_prob_test_cal, threshold=0.5)

        print("\nRain metrics:")
        print(f"Precision: {rain_precision:.4f}")
        print(f"Recall:    {rain_recall:.4f}")
        print(f"F1:        {rain_f1:.4f}")

        latest_rain_window = X_rain_test[-1:].copy() if len(X_rain_test) > 0 else X_rain_val[-1:].copy()
        latest_rain_raw = float(rain_model.predict(latest_rain_window, verbose=0)[0, 0])
        latest_rain_cal = float(calibrator.transform([latest_rain_raw])[0])

        print("\nLatest rain forecast:")
        print(f"Raw rain probability:        {latest_rain_raw:.4f}")
        print(f"Calibrated rain probability: {latest_rain_cal:.4f}")

        rain_diag_plot = plot_rain_diag(y_rain_test, rain_prob_test_cal, threshold=0.5)
        print(f"Saved rain plot: {rain_diag_plot}")

        trained_rain_model = rain_model
        trained_calibrator = calibrator
    else:
        print("\nSkipping rain model.")
        print("Reason: not enough positive rain events in train/validation split.")
        print("Collect more rainy data or reduce RAIN_THRESHOLD / RAIN_HORIZON.")

    print("\nSaved temperature plots:")
    print(temp_loss_plot)
    print(temp_mae_plot)
    print(temp_forecast_plot)

    # -------------------------------------------------------------------
    # GENERATE OUTPUT CSVs
    # -------------------------------------------------------------------
    forecast_csv = generate_forecast_csv(
        df_scaled=df_scaled,
        df_original=df,
        feature_cols=feature_cols,
        lookback=LOOKBACK,
        forecast_steps=FORECAST_WINDOW,
        temp_model=temp_model,
        rain_model=trained_rain_model,
        calibrator=trained_calibrator,
        feature_scaler=feature_scaler,
        output_path="forecast.csv",
    )
    print(f"\nSaved forecast CSV ({FORECAST_WINDOW * 5} min window): {forecast_csv}")

    wind_csv = generate_wind_csv(df, period_hours=24, output_path="wind_average.csv")
    print(f"Saved wind vector-average CSV: {wind_csv}")