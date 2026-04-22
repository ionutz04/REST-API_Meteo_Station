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
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PATTERN = "sensor:264041591600404:*"

BUCKET_SIZE_MSEC = 5 * 60 * 1000

LOOKBACK = 48              # 48 x 5 min = 4 hours, easier baseline
TEMP_HORIZON = 1           # start with 5-minute ahead temp forecast
RAIN_HORIZON = 12          # 1 hour rain event horizon
RAIN_THRESHOLD = 0.1
FORECAST_WINDOW = 60       # 60 x 5 min = 5 hours

# -------------------------------------------------------------------
# PRESSURE-DRIVEN RAIN PROBABILITY HEURISTIC
# -------------------------------------------------------------------
# Each entry: (window_hours, saturation_drop_rate_hpa_per_hour, weight)
#   - window_hours: how far back to look
#   - saturation_drop_rate_hpa_per_hour: drop rate at which P(rain)=~0.95 for
#     that window. Faster windows saturate at higher rates because rapid
#     short-term drops indicate imminent storms.
#   - weight: contribution when combining windows via weighted-max
#
# Reference rules of thumb (hPa per window):
#   - >3 hPa drop in 3h  => imminent storm
#   - >4 hPa drop in 12h => approaching low-pressure system
#   - >5 hPa drop in 24h => sustained deterioration
PRESSURE_WINDOWS_HOURS = [
    # (hours, hPa/h saturation, weight)
    (3.0,  1.0, 1.0),   # short-term / nowcasting
    (12.0, 0.35, 0.9),  # general forecast
    (24.0, 0.22, 0.7),  # long-term trend
]

# Humidity booster (configurable per call).
HUMIDITY_BOOST_THRESHOLD = 70.0   # below this no boost is applied
HUMIDITY_BOOST_WEIGHT = 0.20      # max boost amount when humidity = 100%

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

MIN_RAIN_POSITIVES_TRAIN = 5
MIN_RAIN_POSITIVES_VAL = 2

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
def _clip_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Clip values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - k * iqr, upper=q3 + k * iqr)


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

    # --- NOISE REDUCTION ---
    # 1) IQR-based outlier clipping (caps extreme sensor spikes)
    clip_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    for col in clip_cols:
        df[col] = _clip_outliers_iqr(df[col], k=3.0)

    # 2) Median filter (kernel=3) removes single-sample spikes while
    #    preserving step edges better than a mean filter.
    median_cols = ["temperature", "humidity", "pressure"]
    for col in median_cols:
        df[col] = df[col].rolling(3, min_periods=1, center=True).median()

    # 3) Light exponential moving average to smooth residual jitter.
    #    span=5 → α ≈ 0.33, mild smoothing with low lag.
    ema_cols = ["temperature", "humidity", "pressure"]
    for col in ema_cols:
        df[col] = df[col].ewm(span=5, adjust=False).mean()

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


TARGET_COLS = ["temperature", "humidity", "pressure"]
TARGET_OUT_NAMES = [f"{t}_target" for t in TARGET_COLS]


def make_targets(df: pd.DataFrame, horizon: int = 1):
    """
    Build *parallel* future targets for temperature, humidity, and pressure
    at horizon steps ahead. One row per timestep, three target columns.
    """
    df = df.copy()
    n = len(df)

    for src, tgt in zip(TARGET_COLS, TARGET_OUT_NAMES):
        values = df[src].values
        future = np.full(n, np.nan, dtype=float)
        if n > horizon:
            future[: n - horizon] = values[horizon:]
        df[tgt] = future

    df = df.dropna(subset=TARGET_OUT_NAMES).reset_index(drop=True)
    return df


def _pressure_drop_rate_hpa_per_hour(
    pressure: pd.Series,
    timestamps: pd.Series,
    window_hours: float,
) -> float:
    """
    Average pressure drop rate over the most recent *window_hours* hours.
    Positive value = falling pressure (potential rain).
    Returns None if not enough data covers the window.
    """
    if len(pressure) < 2 or len(timestamps) < 2:
        return None

    last_ts = timestamps.iloc[-1]
    cutoff = last_ts - pd.Timedelta(hours=window_hours)
    mask = timestamps >= cutoff

    p_window = pressure[mask]
    t_window = timestamps[mask]
    if len(p_window) < 2:
        return None

    span_hours = (t_window.iloc[-1] - t_window.iloc[0]).total_seconds() / 3600.0
    if span_hours <= 0:
        return None

    # Require at least 60% of the window covered to trust the signal.
    if span_hours < 0.6 * window_hours:
        return None

    drop_hpa = float(p_window.iloc[0] - p_window.iloc[-1])
    return drop_hpa / span_hours


def _rate_to_probability(drop_rate_hpa_per_h: float, saturation_rate: float) -> float:
    """
    Map a drop rate to a per-window rain probability in [0.05, 0.95].
      - rate <= 0 (rising / flat) => 0.05
      - rate >= saturation_rate   => 0.95
      - linear ramp in between
    """
    if drop_rate_hpa_per_h <= 0.0:
        return 0.05
    score = min(drop_rate_hpa_per_h / saturation_rate, 1.0)
    return 0.05 + 0.90 * score


def estimate_rain_probability_from_pressure(
    df_original: pd.DataFrame,
    pressure_windows=PRESSURE_WINDOWS_HOURS,
    humidity_boost_weight: float = HUMIDITY_BOOST_WEIGHT,
    humidity_boost_threshold: float = HUMIDITY_BOOST_THRESHOLD,
) -> float:
    """
    Estimate current rain probability from multi-window pressure drop rates,
    optionally boosted by recent humidity.

    Methodology:
      1. For each (window_hours, saturation_rate, weight), compute the
         average pressure drop rate (hPa/hour) and convert it to a per-window
         probability via _rate_to_probability.
      2. Combine windows with weighted-max: the most alarming signal wins,
         but each is scaled by its weight to reflect confidence.
      3. Apply optional humidity boost above *humidity_boost_threshold*,
         scaled linearly up to *humidity_boost_weight* at 100% RH.
    """
    if "pressure" not in df_original.columns or "timestamp" not in df_original.columns:
        return 0.5

    df = df_original[["timestamp", "pressure"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["pressure"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) < 2:
        return 0.5

    pressure = pd.to_numeric(df["pressure"], errors="coerce")
    timestamps = df["timestamp"]

    weighted_probs = []
    for window_hours, saturation_rate, weight in pressure_windows:
        rate = _pressure_drop_rate_hpa_per_hour(pressure, timestamps, window_hours)
        if rate is None:
            continue
        per_window_prob = _rate_to_probability(rate, saturation_rate)
        # Pull non-confident weights toward neutral 0.5, then take max.
        weighted = 0.5 + weight * (per_window_prob - 0.5)
        weighted_probs.append(weighted)

    if not weighted_probs:
        return 0.5

    prob = float(max(weighted_probs))

    # Humidity booster (configurable).
    if humidity_boost_weight > 0.0 and "humidity" in df_original.columns:
        humidity = pd.to_numeric(df_original["humidity"], errors="coerce").dropna()
        if len(humidity) > 0:
            recent_h = float(humidity.tail(36).mean())  # last ~3h at 5-min buckets
            if recent_h > humidity_boost_threshold:
                span = max(1e-6, 100.0 - humidity_boost_threshold)
                boost = humidity_boost_weight * np.clip(
                    (recent_h - humidity_boost_threshold) / span, 0.0, 1.0
                )
                prob = prob + (1.0 - prob) * float(boost)

    return float(np.clip(prob, 0.01, 0.99))


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


def fit_target_scaler_on_train(df, target_cols, train_end_row):
    scaler = StandardScaler()
    scaler.fit(df.loc[:train_end_row - 1, target_cols])
    return scaler


def apply_target_scaler(df, target_cols, scaler):
    df = df.copy()
    df[target_cols] = scaler.transform(df[target_cols]).astype(np.float32)
    return df


def build_multi_target_sequences(df, feature_cols, target_cols, lookback, start_idx, end_idx):
    """
    Build sequences with parallel multi-target outputs:
      X[i]  -> shape (lookback, n_features)
      Y[i]  -> shape (n_targets,)
    Strict chronological windows: start_idx <= target_index < end_idx.
    """
    feature_values = df[feature_cols].values
    target_values = df[target_cols].values
    ts_values = df["timestamp"].values

    X, Y, ts = [], [], []
    effective_start = max(start_idx, lookback)
    for i in range(effective_start, end_idx):
        X.append(feature_values[i - lookback:i])
        Y.append(target_values[i])
        ts.append(ts_values[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(ts),
    )


# -------------------------------------------------------------------
# MODEL: shared-trunk LSTM with parallel heads
# -------------------------------------------------------------------
def build_multi_output_model(n_steps, n_features, n_targets=3):
    """
    Single LSTM trunk -> parallel Dense head producing all targets at once.
    Inputs are parallel multivariate features; outputs are parallel
    regressions (temperature, humidity, pressure for next step).
    """
    inp = Input(shape=(n_steps, n_features), name="weather_window")

    x = LSTM(
        32,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(inp)
    x = Dropout(0.30)(x)
    x = LSTM(
        16,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(x)
    x = Dropout(0.30)(x)
    x = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)

    out = Dense(n_targets, name="future_state")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
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


# -------------------------------------------------------------------
# FORECAST CSV (recursive multi-step using single multi-output LSTM)
# -------------------------------------------------------------------
def generate_forecast_csv(
    df_scaled,
    df_original,
    feature_cols,
    target_cols,
    lookback,
    forecast_steps,
    model,
    feature_scaler,
    target_scaler=None,
    output_path="forecast.csv",
):
    """
    Recursively roll the multi-output LSTM forward step-by-step.
    Each step:
      - feed the last `lookback` window of scaled features to the model
      - obtain predicted next-step temperature, humidity, pressure (parallel)
      - construct the next raw feature row (using these predictions for any
        feature column that is also a target), rescale, and append
      - rain probability is derived from the LSTM-predicted *future pressure
        trajectory* using the same multi-window heuristic.
    """
    last_ts = pd.Timestamp(df_original["timestamp"].iloc[-1])
    bucket_td = pd.Timedelta(minutes=5)

    window = df_scaled[feature_cols].values[-lookback:].copy()
    last_raw = df_original[feature_cols].iloc[-1].values.astype(float).copy()

    hist = df_original.tail(lookback).copy()
    ws = pd.to_numeric(hist["wind_speed"], errors="coerce").fillna(0.0).values
    wd = np.deg2rad(pd.to_numeric(hist["wind_direction_degrees"], errors="coerce").fillna(0.0).values)
    mean_u = np.mean(-ws * np.sin(wd))
    mean_v = np.mean(-ws * np.cos(wd))
    hist_wind_speed = round(float(np.sqrt(mean_u ** 2 + mean_v ** 2)), 2)
    hist_wind_dir = round(float(np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360), 1)

    # Recent observed history (for blending into rain estimation context).
    hist_pressure = pd.to_numeric(df_original["pressure"], errors="coerce").dropna()
    hist_humidity = pd.to_numeric(df_original["humidity"], errors="coerce").dropna()
    hist_temperature = pd.to_numeric(df_original["temperature"], errors="coerce").dropna()
    hist_timestamps = pd.to_datetime(df_original["timestamp"]).reset_index(drop=True)

    target_to_feature_idx = {t: feature_cols.index(t) for t in target_cols if t in feature_cols}

    pred_temps, pred_hums, pred_press = [], [], []
    pred_timestamps = []

    rows = []
    for step in range(1, forecast_steps + 1):
        inp = window[np.newaxis, :, :]
        preds = model.predict(inp, verbose=0)[0]  # shape (n_targets,)
        if target_scaler is not None:
            preds = target_scaler.inverse_transform(preds.reshape(1, -1))[0]
        pred_map = {name: float(val) for name, val in zip(target_cols, preds)}

        forecast_ts = last_ts + bucket_td * step
        pred_timestamps.append(forecast_ts)
        pred_temps.append(pred_map.get("temperature", np.nan))
        pred_hums.append(pred_map.get("humidity", np.nan))
        pred_press.append(pred_map.get("pressure", np.nan))

        # Build rolling pressure/humidity/temperature series = observed + predicted-so-far
        combined_pressure = pd.concat(
            [hist_pressure, pd.Series(pred_press)], ignore_index=True
        )
        combined_humidity = pd.concat(
            [hist_humidity, pd.Series(pred_hums)], ignore_index=True
        )
        combined_temperature = pd.concat(
            [hist_temperature, pd.Series(pred_temps)], ignore_index=True
        )
        combined_timestamps = pd.concat(
            [hist_timestamps, pd.Series(pred_timestamps)], ignore_index=True
        )
        synth_df = pd.DataFrame({
            "timestamp": combined_timestamps,
            "pressure": combined_pressure,
            "humidity": combined_humidity,
            "temperature": combined_temperature,
        })
        rain_prob = estimate_rain_probability_from_pressure(synth_df)

        rows.append({
            "forecast_timestamp": forecast_ts,
            "minutes_ahead": step * 5,
            "predicted_temperature_C": round(pred_map.get("temperature", float("nan")), 2),
            "predicted_humidity_pct": round(pred_map.get("humidity", float("nan")), 2),
            "predicted_pressure_hPa": round(pred_map.get("pressure", float("nan")), 2),
            "rain_probability": round(rain_prob, 4),
            "vector_avg_wind_speed": hist_wind_speed,
            "vector_avg_wind_direction_deg": hist_wind_dir,
        })

        # Build next raw feature row by replacing target columns with predictions.
        new_raw = last_raw.copy()
        for tgt, feat_idx in target_to_feature_idx.items():
            new_raw[feat_idx] = pred_map[tgt]
        new_scaled = feature_scaler.transform(
            pd.DataFrame([new_raw], columns=feature_cols)
        ).flatten()
        window = np.vstack([window[1:], new_scaled])
        last_raw = new_raw

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
    last *period_hours* hours of data. Produces one summary row.
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


def plot_predicted_data_over_entire_series(
    df_scaled,
    df_original,
    feature_cols,
    target_cols,
    lookback,
    model,
    output_dir=OUTPUT_DIR,
    target_scaler=None,
):
    """
    Plot actual vs predicted for each target (temperature, humidity, pressure)
    over the entire valid series (train + val + test) using timestamps.
    """
    path = ensure_output_dir(output_dir)

    X_all, Y_all, ts_all = build_multi_target_sequences(
        df_scaled,
        feature_cols,
        [f"{t}_target" for t in target_cols],
        lookback,
        start_idx=0,
        end_idx=len(df_scaled),
    )
    if len(X_all) == 0:
        raise RuntimeError("No sequences available to plot over the entire series.")

    Y_pred = model.predict(X_all, verbose=0)
    if target_scaler is not None:
        Y_all = target_scaler.inverse_transform(Y_all)
        Y_pred = target_scaler.inverse_transform(Y_pred)

    units = {"temperature": "deg C", "humidity": "%", "pressure": "hPa"}
    out_paths = []
    for i, target in enumerate(target_cols):
        fig = plt.figure(figsize=(15, 5.5))
        plt.plot(ts_all, Y_all[:, i], label=f"actual_{target}", linewidth=1.8)
        plt.plot(ts_all, Y_pred[:, i], label=f"pred_{target}", linewidth=1.4, alpha=0.85)
        plt.title(f"Predicted {target.capitalize()} Over Entire Series")
        plt.xlabel("Timestamp")
        plt.ylabel(f"{target.capitalize()} ({units.get(target, '')})")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out = path / f"{target}_entire_series_prediction.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(str(out))
    return out_paths


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


def plot_temp_forecast(y_true, y_pred, target_name, output_dir=OUTPUT_DIR):
    path = ensure_output_dir(output_dir)
    fig = plt.figure(figsize=(14, 5))
    plt.plot(y_true, label=f"actual_{target_name}", linewidth=2)
    plt.plot(y_pred, label=f"pred_{target_name}", linewidth=2)
    plt.title(f"{target_name.capitalize()} Forecast on Test Set")
    plt.xlabel("Test sample index")
    plt.ylabel(target_name.capitalize())
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = path / f"{target_name}_forecast_test.png"
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
    df = make_targets(df, horizon=TEMP_HORIZON)

    print("Loaded rows after feature/target prep:", len(df))
    print("\nTarget summaries:")
    print(df[TARGET_OUT_NAMES].describe())

    current_pressure_rain_prob = estimate_rain_probability_from_pressure(df)
    print("\nPressure-based rain probability estimate (current observed):")
    print(f"P(rain): {current_pressure_rain_prob:.3f}")

    feature_cols = [
        "temperature",
        "humidity",
        "pressure",
    ]
    target_cols = TARGET_COLS
    target_out_names = TARGET_OUT_NAMES

    train_end_row, val_end_row = chronological_split_indices(len(df), TRAIN_RATIO, VAL_RATIO)

    feature_scaler = fit_feature_scaler_on_train(df, feature_cols, train_end_row)
    df_scaled = apply_feature_scaler(df, feature_cols, feature_scaler)

    target_scaler = fit_target_scaler_on_train(df_scaled, target_out_names, train_end_row)
    df_scaled = apply_target_scaler(df_scaled, target_out_names, target_scaler)

    # -------------------------------
    # SINGLE MULTI-OUTPUT LSTM
    # -------------------------------
    X_train, Y_train, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, 0, train_end_row,
    )
    X_val, Y_val, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, train_end_row, val_end_row,
    )
    X_test, Y_test, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, val_end_row, len(df_scaled),
    )

    print("\nMulti-output sequence shapes:")
    print("train:", X_train.shape, Y_train.shape)
    print("val:  ", X_val.shape, Y_val.shape)
    print("test: ", X_test.shape, Y_test.shape)

    model = build_multi_output_model(
        n_steps=X_train.shape[1],
        n_features=X_train.shape[2],
        n_targets=Y_train.shape[1],
    )

    early_stop = EarlyStopping(
        monitor="val_mae",
        patience=8,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        shuffle=False,
        verbose=1,
    )

    Y_pred_test_scaled = model.predict(X_test, verbose=0)
    Y_test_real = target_scaler.inverse_transform(Y_test)
    Y_pred_test = target_scaler.inverse_transform(Y_pred_test_scaled)

    print("\nPer-target test metrics (real units):")
    for i, name in enumerate(target_cols):
        mae, rmse = evaluate_temperature(Y_test_real[:, i], Y_pred_test[:, i])
        print(f"  {name:12s}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    print("\nLatest one-step predictions (real units):")
    latest_window = X_test[-1:].copy() if len(X_test) > 0 else X_val[-1:].copy()
    latest_pred_scaled = model.predict(latest_window, verbose=0)
    latest_pred = target_scaler.inverse_transform(latest_pred_scaled)[0]
    for name, val in zip(target_cols, latest_pred):
        print(f"  {name}: {val:.3f}")

    loss_plot = plot_temp_training(history, output_dir=OUTPUT_DIR)
    mae_plot = plot_temp_mae(history, output_dir=OUTPUT_DIR)
    forecast_plots = [
        plot_temp_forecast(Y_test_real[:, i], Y_pred_test[:, i], target_cols[i], output_dir=OUTPUT_DIR)
        for i in range(len(target_cols))
    ]
    entire_series_plots = plot_predicted_data_over_entire_series(
        df_scaled=df_scaled,
        df_original=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        lookback=LOOKBACK,
        model=model,
        output_dir=OUTPUT_DIR,
        target_scaler=target_scaler,
    )

    print("\nSaved plots:")
    print(loss_plot)
    print(mae_plot)
    for p in forecast_plots:
        print(p)
    for p in entire_series_plots:
        print(p)

    # -------------------------------------------------------------------
    # GENERATE OUTPUT CSVs
    # -------------------------------------------------------------------
    forecast_csv = generate_forecast_csv(
        df_scaled=df_scaled,
        df_original=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        lookback=LOOKBACK,
        forecast_steps=FORECAST_WINDOW,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        output_path="forecast.csv",
    )
    print(f"\nSaved forecast CSV ({FORECAST_WINDOW * 5} min window): {forecast_csv}")

    wind_csv = generate_wind_csv(df, period_hours=0.17, output_path="wind_average.csv")
    print(f"Saved wind vector-average CSV: {wind_csv}")