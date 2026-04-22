"""
Multi-head forecasting variant.

Architecture:
  * SHARED LSTM TRUNK on the recent sensor window (T, H, P + derivatives).
  * TEMPERATURE HEAD: trunk -> Dense, augmented with cyclical seasonality
    (hour_sin/cos, doy_sin/cos). Temperature is the strongest cyclic
    variable so it benefits the most from explicit seasonal priors.
  * HUMIDITY HEAD:    trunk -> Dense, augmented with the temperature head's
    OWN prediction. Humidity is anti-correlated with temperature, so giving
    the humidity head the predicted temperature lets it exploit that link
    instead of re-learning it.
  * PRESSURE MODEL:   *NOT* an LSTM. A Ridge linear regressor on lagged
    pressure differences (Delta P over 30 min / 1 h / 3 h / 6 h) plus
    cyclical hour features. Pressure behaves like a slowly drifting random
    walk; a simple linear extrapolator is competitive and far less prone to
    drifting toward a learned mean than an LSTM.

Pipeline:
  1. Load Redis sensor data and prep features (re-uses predict_rain helpers).
  2. Build sequences for the LSTM (temperature + humidity targets).
  3. Build a tabular dataset for the linear pressure model.
  4. Train both, evaluate per-target on the held-out test slice.
  5. Produce the standard forecast.csv + new diagnostic plots:
       - pressure_residuals.png
       - pressure_linear_coeffs.png
       - humidity_vs_temp_scatter.png
       - per-target forecast_test plots
       - per-target entire_series plots

Outputs:
  forecast.csv              (5h forward forecast, 5-min cadence)
  wind_average.csv          (vector-averaged wind, mirrors predict_rain.py)
  debug_plots/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from predict_rain import (
    REDIS_PATTERN,
    LOOKBACK,
    TEMP_HORIZON,
    FORECAST_WINDOW,
    TRAIN_RATIO,
    VAL_RATIO,
    OUTPUT_DIR,
    RANDOM_SEED,
    get_data_from_redis,
    prepare_features,
    make_targets,
    estimate_rain_probability_from_pressure,
    chronological_split_indices,
    fit_feature_scaler_on_train,
    apply_feature_scaler,
    fit_target_scaler_on_train,
    apply_target_scaler,
    build_multi_target_sequences,
    generate_wind_csv,
)


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
LSTM_TARGET_COLS = ["temperature", "humidity"]
LSTM_TARGET_OUT_NAMES = [f"{t}_target" for t in LSTM_TARGET_COLS]

PRESSURE_LAG_MINUTES = [30, 60, 180, 360]   # Delta-P windows for the linear model
PRESSURE_RIDGE_ALPHA = 1.0

# 2 seasonality columns for the temperature head (hour-of-day only).
# day-of-year features were removed: with only ~weeks of training data
# they're effectively constant and the model latched onto coincidental
# correlations with them, biasing test-time predictions warm.
SEASONALITY_COLS = ["hour_sin", "hour_cos"]

# Probability of zeroing the seasonality side-input during training.
# Forces the temperature head to also lean on the LSTM trunk's recent
# state instead of blindly trusting time-of-day.
SEASONALITY_DROPOUT = 0.3

LSTM_EPOCHS = 80
LSTM_BATCH_SIZE = 32
LSTM_LR = 1e-3

# Recency weighting (same idea as in predict_rain_with_transfer_learing.py):
# the most-recent N% of training samples are weighed higher so the model
# adapts to the current weather regime instead of averaging over the
# possibly-different older period.
RECENT_FRACTION = 0.25
RECENT_WEIGHT = 3.0

OUTPUT_DIR_MULTIHEAD = "debug_plots_multihead"
OUTPUT_DIR_PATH = Path(OUTPUT_DIR_MULTIHEAD)
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# -------------------------------------------------------------------
# SEASONALITY
# -------------------------------------------------------------------
def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


def seasonality_row(ts: pd.Timestamp) -> np.ndarray:
    hour = ts.hour + ts.minute / 60.0
    return np.array([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
    ], dtype=np.float32)


# -------------------------------------------------------------------
# LSTM MODEL: shared trunk + (temp, humidity) heads
# -------------------------------------------------------------------
def build_multihead_lstm(n_steps: int, n_features: int) -> Model:
    seq_in = Input(shape=(n_steps, n_features), name="sequence_input")
    season_in = Input(shape=(len(SEASONALITY_COLS),), name="seasonality_input")

    x = LSTM(64, return_sequences=True)(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    trunk = Dropout(0.2)(x)

    # Dropout on the seasonality side-input forces the temperature head to
    # also rely on the LSTM trunk instead of blindly trusting time-of-day.
    season_in_drop = Dropout(SEASONALITY_DROPOUT)(season_in)

    # Temperature head: trunk + (noisy) seasonality
    t_in = Concatenate()([trunk, season_in_drop])
    t = Dense(32, activation="relu")(t_in)
    temp_out = Dense(1, name="temperature_out")(t)

    # Humidity head: trunk + predicted temperature
    h_in = Concatenate()([trunk, temp_out])
    h = Dense(32, activation="relu")(h_in)
    hum_out = Dense(1, name="humidity_out")(h)

    model = Model(inputs=[seq_in, season_in], outputs=[temp_out, hum_out])
    return model


# -------------------------------------------------------------------
# LINEAR PRESSURE MODEL
# -------------------------------------------------------------------
def build_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the design matrix for the linear pressure model.
    Features: current pressure + Delta P over a few lag windows + cyclical hour.
    Target:   pressure at t + TEMP_HORIZON  (already in df as 'pressure_target'
              if make_targets was called; otherwise we build it here).
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    if "pressure_target" not in df.columns:
        n = len(df)
        future = np.full(n, np.nan, dtype=float)
        if n > TEMP_HORIZON:
            future[: n - TEMP_HORIZON] = df["pressure"].values[TEMP_HORIZON:]
        df["pressure_target"] = future

    # 5-min cadence -> N-step lag for N minutes
    cadence = 5
    feats = {"pressure": df["pressure"].astype(float)}
    for m in PRESSURE_LAG_MINUTES:
        steps = max(1, m // cadence)
        feats[f"dp_{m}m"] = df["pressure"].astype(float) - df["pressure"].astype(float).shift(steps)

    # cyclical hour
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    out = pd.DataFrame(feats)
    out["timestamp"] = df["timestamp"].values
    out["pressure_target"] = df["pressure_target"].values
    out = out.dropna().reset_index(drop=True)
    return out


def train_pressure_model(p_df: pd.DataFrame, train_end: int, val_end: int):
    feat_cols = [c for c in p_df.columns if c not in ("timestamp", "pressure_target")]
    X_tr = p_df.loc[: train_end - 1, feat_cols].values
    y_tr = p_df.loc[: train_end - 1, "pressure_target"].values
    X_va = p_df.loc[train_end : val_end - 1, feat_cols].values
    y_va = p_df.loc[train_end : val_end - 1, "pressure_target"].values
    X_te = p_df.loc[val_end:, feat_cols].values
    y_te = p_df.loc[val_end:, "pressure_target"].values
    ts_te = p_df.loc[val_end:, "timestamp"].values

    scaler = StandardScaler().fit(X_tr)
    model = Ridge(alpha=PRESSURE_RIDGE_ALPHA).fit(scaler.transform(X_tr), y_tr)

    val_pred = model.predict(scaler.transform(X_va))
    test_pred = model.predict(scaler.transform(X_te))
    print(f"[pressure] val  MAE={mean_absolute_error(y_va, val_pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_va, val_pred)):.4f}")
    print(f"[pressure] test MAE={mean_absolute_error(y_te, test_pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_te, test_pred)):.4f}")
    return {
        "model": model,
        "scaler": scaler,
        "feat_cols": feat_cols,
        "y_te": y_te,
        "pred_te": test_pred,
        "ts_te": ts_te,
    }


def predict_pressure_one_step(p_model: dict, hist_pressure: list, ts: pd.Timestamp) -> float:
    """
    Compute one forward pressure prediction from a rolling pressure history
    (list of floats). The history must be long enough to cover the largest lag.
    """
    cadence = 5
    cur = float(hist_pressure[-1])
    feats = [cur]
    for m in PRESSURE_LAG_MINUTES:
        steps = max(1, m // cadence)
        if len(hist_pressure) > steps:
            feats.append(cur - float(hist_pressure[-steps - 1]))
        else:
            feats.append(0.0)
    hour = ts.hour + ts.minute / 60.0
    feats.append(float(np.sin(2 * np.pi * hour / 24.0)))
    feats.append(float(np.cos(2 * np.pi * hour / 24.0)))

    X = p_model["scaler"].transform(np.array(feats, dtype=float).reshape(1, -1))
    return float(p_model["model"].predict(X)[0])


# -------------------------------------------------------------------
# DIAGNOSTIC PLOTS
# -------------------------------------------------------------------
def plot_pressure_residuals(p_model: dict, output_dir: str = OUTPUT_DIR_MULTIHEAD) -> str:
    y, p, ts = p_model["y_te"], p_model["pred_te"], pd.to_datetime(p_model["ts_te"])
    res = y - p
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(ts, y, label="actual", color="tab:blue")
    axes[0].plot(ts, p, label="linear pred", color="tab:orange", alpha=0.85)
    axes[0].set_ylabel("Pressure (hPa)")
    axes[0].set_title("Pressure: linear model vs. actual (test slice)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ts, res, color="tab:red", linewidth=0.9)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Residual (hPa)")
    axes[1].set_xlabel("time")
    axes[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    out = Path(output_dir) / "pressure_residuals.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_pressure_coeffs(p_model: dict, output_dir: str = OUTPUT_DIR_MULTIHEAD) -> str:
    coefs = p_model["model"].coef_
    names = p_model["feat_cols"]
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(np.abs(coefs))[::-1]
    ax.barh([names[i] for i in order][::-1],
            [coefs[i] for i in order][::-1],
            color="teal")
    ax.set_title("Ridge pressure model: standardised coefficients")
    ax.set_xlabel("coef")
    ax.grid(axis="x", alpha=0.3)
    out = Path(output_dir) / "pressure_linear_coeffs.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_humidity_vs_temp(temp_pred, hum_pred, hum_actual, output_dir: str = OUTPUT_DIR_MULTIHEAD) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(temp_pred, hum_pred, c=hum_actual, cmap="viridis",
                    s=18, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="actual humidity (%)")
    ax.set_xlabel("predicted temperature (°C)")
    ax.set_ylabel("predicted humidity (%)")
    ax.set_title("Humidity vs Temperature predictions (color = actual humidity)")
    ax.grid(alpha=0.3)
    out = Path(output_dir) / "humidity_vs_temp_scatter.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_target_test(actual, pred, name, output_dir: str = OUTPUT_DIR_MULTIHEAD) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(actual, label="actual", color="tab:blue")
    ax.plot(pred, label="predicted", color="tab:orange", alpha=0.9)
    ax.set_title(f"{name} - test forecast (one-step)")
    ax.set_xlabel("test sample")
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path(output_dir) / f"{name}_forecast_test.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_entire_series(actual, pred, name, output_dir: str = OUTPUT_DIR_MULTIHEAD) -> str:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(actual, label="actual", color="tab:blue")
    ax.plot(pred, label="predicted", color="tab:orange", alpha=0.85)
    ax.set_title(f"{name} - one-step prediction over entire series")
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path(output_dir) / f"{name}_entire_series_prediction.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_lstm_loss(history, output_dir: str = OUTPUT_DIR_MULTIHEAD) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in history.history:
        if "loss" in k:
            ax.plot(history.history[k], label=k)
    ax.set_title("LSTM training loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(alpha=0.3)
    out = Path(output_dir) / "training_loss.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# -------------------------------------------------------------------
# FORECAST CSV (recursive roll-out, mixing LSTM heads + linear pressure)
# -------------------------------------------------------------------
def generate_forecast_csv_multihead(
    df_original: pd.DataFrame,
    df_scaled: pd.DataFrame,
    feature_cols: list,
    lookback: int,
    forecast_steps: int,
    lstm_model: Model,
    feature_scaler,
    target_scaler,
    pressure_model: dict,
    output_path: str = "forecast.csv",
):
    last_ts = pd.Timestamp(df_original["timestamp"].iloc[-1])
    bucket_td = pd.Timedelta(minutes=5)

    window = df_scaled[feature_cols].values[-lookback:].copy()
    last_raw = df_original[feature_cols].iloc[-1].values.astype(float).copy()

    # Pressure history needs enough back-data for the largest lag.
    max_lag_steps = max(PRESSURE_LAG_MINUTES) // 5 + 2
    hist_pressure = list(
        df_original["pressure"].astype(float).tail(max_lag_steps).values
    )

    # For wind row (kept constant over the forecast horizon)
    hist = df_original.tail(lookback).copy()
    ws = pd.to_numeric(hist["wind_speed"], errors="coerce").fillna(0.0).values
    wd = np.deg2rad(pd.to_numeric(hist["wind_direction_degrees"], errors="coerce").fillna(0.0).values)
    mean_u = np.mean(-ws * np.sin(wd))
    mean_v = np.mean(-ws * np.cos(wd))
    hist_wind_speed = round(float(np.sqrt(mean_u ** 2 + mean_v ** 2)), 2)
    hist_wind_dir = round(float(np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360), 1)

    hist_pressure_full = pd.to_numeric(df_original["pressure"], errors="coerce").dropna()
    hist_humidity_full = pd.to_numeric(df_original["humidity"], errors="coerce").dropna()
    hist_temperature_full = pd.to_numeric(df_original["temperature"], errors="coerce").dropna()
    hist_timestamps = pd.to_datetime(df_original["timestamp"]).reset_index(drop=True)

    # Where target columns sit inside feature_cols (so we can write back the
    # next step's predictions and keep rolling).
    feat_idx_temp = feature_cols.index("temperature") if "temperature" in feature_cols else None
    feat_idx_hum = feature_cols.index("humidity") if "humidity" in feature_cols else None
    feat_idx_press = feature_cols.index("pressure") if "pressure" in feature_cols else None
    season_feat_idx = {c: feature_cols.index(c) for c in SEASONALITY_COLS if c in feature_cols}

    pred_temps, pred_hums, pred_press, pred_ts = [], [], [], []
    rows = []

    for step in range(1, forecast_steps + 1):
        forecast_ts = last_ts + bucket_td * step
        season = seasonality_row(forecast_ts)

        seq_in = window[np.newaxis, :, :]
        season_in = season[np.newaxis, :]
        t_pred_scaled, h_pred_scaled = lstm_model.predict([seq_in, season_in], verbose=0)
        # invert target scaler (it was fitted on [temperature, humidity])
        scaled = np.array([[t_pred_scaled[0, 0], h_pred_scaled[0, 0]]], dtype=np.float32)
        unscaled = target_scaler.inverse_transform(scaled)[0]
        t_pred = float(unscaled[0])
        h_pred = float(np.clip(unscaled[1], 0.0, 100.0))

        # Pressure: linear model over the rolling pressure history.
        p_pred = predict_pressure_one_step(pressure_model, hist_pressure, forecast_ts)
        hist_pressure.append(p_pred)
        if len(hist_pressure) > max_lag_steps + 5:
            hist_pressure = hist_pressure[-(max_lag_steps + 5):]

        pred_temps.append(t_pred); pred_hums.append(h_pred)
        pred_press.append(p_pred); pred_ts.append(forecast_ts)

        synth_df = pd.DataFrame({
            "timestamp": pd.concat(
                [hist_timestamps, pd.Series(pred_ts)], ignore_index=True),
            "pressure": pd.concat(
                [hist_pressure_full, pd.Series(pred_press)], ignore_index=True),
            "humidity": pd.concat(
                [hist_humidity_full, pd.Series(pred_hums)], ignore_index=True),
            "temperature": pd.concat(
                [hist_temperature_full, pd.Series(pred_temps)], ignore_index=True),
        })
        rain_prob = estimate_rain_probability_from_pressure(synth_df)

        rows.append({
            "forecast_timestamp": forecast_ts,
            "minutes_ahead": step * 5,
            "predicted_temperature_C": round(t_pred, 2),
            "predicted_humidity_pct": round(h_pred, 2),
            "predicted_pressure_hPa": round(p_pred, 2),
            "rain_probability": round(rain_prob, 4),
            "vector_avg_wind_speed": hist_wind_speed,
            "vector_avg_wind_direction_deg": hist_wind_dir,
        })

        # Roll the LSTM input window forward.
        new_raw = last_raw.copy()
        if feat_idx_temp is not None:
            new_raw[feat_idx_temp] = t_pred
        if feat_idx_hum is not None:
            new_raw[feat_idx_hum] = h_pred
        if feat_idx_press is not None:
            new_raw[feat_idx_press] = p_pred
        next_ts = forecast_ts + bucket_td
        next_season = seasonality_row(next_ts)
        for col, fi in season_feat_idx.items():
            new_raw[fi] = next_season[SEASONALITY_COLS.index(col)]

        new_scaled = feature_scaler.transform(
            pd.DataFrame([new_raw], columns=feature_cols)
        ).flatten()
        window = np.vstack([window[1:], new_scaled])
        last_raw = new_raw

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    return str(output_path)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def run() -> None:
    feature_cols = ["temperature", "humidity", "pressure"] + SEASONALITY_COLS

    df = get_data_from_redis()
    if df.empty:
        raise RuntimeError(f"No data loaded from Redis for pattern {REDIS_PATTERN}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = prepare_features(df)
    df = add_seasonality(df)
    df_with_targets = make_targets(df, horizon=TEMP_HORIZON)
    # make_targets only produces the original 3 targets; we rebuild for the
    # subset we care about for the LSTM (temperature + humidity).
    df_with_targets = df_with_targets[
        ["timestamp", *feature_cols, "wind_speed", "wind_direction_degrees", "rainfall",
         "temperature_target", "humidity_target", "pressure_target"]
    ].copy()
    print(f"[main] Local rows after prep: {len(df_with_targets)}")

    train_end, val_end = chronological_split_indices(
        len(df_with_targets), TRAIN_RATIO, VAL_RATIO,
    )

    # ---- LSTM data prep (temp + humidity) ----
    feat_scaler = fit_feature_scaler_on_train(df_with_targets, feature_cols, train_end)
    df_scaled = apply_feature_scaler(df_with_targets, feature_cols, feat_scaler)
    targ_scaler = fit_target_scaler_on_train(df_scaled, LSTM_TARGET_OUT_NAMES, train_end)
    df_scaled = apply_target_scaler(df_scaled, LSTM_TARGET_OUT_NAMES, targ_scaler)

    X_tr, Y_tr, _ = build_multi_target_sequences(
        df_scaled, feature_cols, LSTM_TARGET_OUT_NAMES, LOOKBACK, 0, train_end)
    X_va, Y_va, _ = build_multi_target_sequences(
        df_scaled, feature_cols, LSTM_TARGET_OUT_NAMES, LOOKBACK, train_end, val_end)
    X_te, Y_te, _ = build_multi_target_sequences(
        df_scaled, feature_cols, LSTM_TARGET_OUT_NAMES, LOOKBACK, val_end, len(df_scaled))
    print(f"[lstm] sequences  train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")

    season_idx = [feature_cols.index(c) for c in SEASONALITY_COLS]
    # The seasonality side-input is the seasonality at the *current step*
    # being predicted -> last row of each input window.
    S_tr = X_tr[:, -1, season_idx]
    S_va = X_va[:, -1, season_idx]
    S_te = X_te[:, -1, season_idx]

    Y_tr_t, Y_tr_h = Y_tr[:, 0:1], Y_tr[:, 1:2]
    Y_va_t, Y_va_h = Y_va[:, 0:1], Y_va[:, 1:2]
    Y_te_t, Y_te_h = Y_te[:, 0:1], Y_te[:, 1:2]

    # ---- Build + train LSTM ----
    model = build_multihead_lstm(n_steps=X_tr.shape[1], n_features=X_tr.shape[2])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss={"temperature_out": "mse", "humidity_out": "mse"},
        metrics={"temperature_out": "mae", "humidity_out": "mae"},
    )
    es = EarlyStopping(monitor="val_loss", patience=10,
                       restore_best_weights=True, verbose=1)

    # Recency-weighted training: emphasise the most-recent slice so the
    # model adapts to current conditions instead of averaging history.
    sw = np.ones(len(X_tr), dtype=np.float32)
    if RECENT_FRACTION > 0 and len(sw) > 0:
        n_recent = max(1, int(round(len(sw) * RECENT_FRACTION)))
        sw[-n_recent:] = RECENT_WEIGHT
        print(f"[lstm] sample_weight: last {n_recent}/{len(sw)} samples "
              f"@ x{RECENT_WEIGHT}")

    history = model.fit(
        [X_tr, S_tr], [Y_tr_t, Y_tr_h],
        sample_weight=[sw, sw],
        validation_data=([X_va, S_va], [Y_va_t, Y_va_h]),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=[es],
        shuffle=False,
        verbose=1,
    )

    # ---- LSTM evaluation (real units) ----
    t_pred_s, h_pred_s = model.predict([X_te, S_te], verbose=0)
    pred_scaled = np.hstack([t_pred_s, h_pred_s])
    pred_real = targ_scaler.inverse_transform(pred_scaled)
    actual_real = targ_scaler.inverse_transform(Y_te)
    for i, name in enumerate(LSTM_TARGET_COLS):
        mae = mean_absolute_error(actual_real[:, i], pred_real[:, i])
        rmse = np.sqrt(mean_squared_error(actual_real[:, i], pred_real[:, i]))
        print(f"[lstm] {name:11s} test MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ---- Linear pressure model ----
    p_df = build_pressure_features(df_with_targets)
    p_train_end, p_val_end = chronological_split_indices(
        len(p_df), TRAIN_RATIO, VAL_RATIO)
    p_model = train_pressure_model(p_df, p_train_end, p_val_end)

    # ---- Plots ----
    plots = []
    plots.append(plot_lstm_loss(history))
    for i, name in enumerate(LSTM_TARGET_COLS):
        plots.append(plot_target_test(actual_real[:, i], pred_real[:, i], name))
    plots.append(plot_target_test(p_model["y_te"], p_model["pred_te"], "pressure"))

    # entire-series one-step plots (LSTM heads + pressure)
    full_t_s, full_h_s = model.predict(
        [_full_sequences(df_scaled, feature_cols, LOOKBACK),
         _full_seasonality(df_scaled, feature_cols, LOOKBACK)],
        verbose=0,
    )
    full_pred = targ_scaler.inverse_transform(np.hstack([full_t_s, full_h_s]))
    actual_full_t = df_with_targets["temperature_target"].values[LOOKBACK:]
    actual_full_h = df_with_targets["humidity_target"].values[LOOKBACK:]
    plots.append(plot_entire_series(actual_full_t, full_pred[:, 0], "temperature"))
    plots.append(plot_entire_series(actual_full_h, full_pred[:, 1], "humidity"))

    # full pressure prediction over entire series
    feat_cols_p = p_model["feat_cols"]
    X_full_p = p_model["scaler"].transform(p_df[feat_cols_p].values)
    full_p_pred = p_model["model"].predict(X_full_p)
    plots.append(plot_entire_series(p_df["pressure_target"].values, full_p_pred, "pressure"))

    # diagnostics specific to multihead
    plots.append(plot_pressure_residuals(p_model))
    plots.append(plot_pressure_coeffs(p_model))
    plots.append(plot_humidity_vs_temp(pred_real[:, 0], pred_real[:, 1], actual_real[:, 1]))

    print("\nSaved plots:")
    for p in plots:
        print(p)

    # ---- Forecast CSV ----
    fcsv = generate_forecast_csv_multihead(
        df_original=df_with_targets,
        df_scaled=df_scaled,
        feature_cols=feature_cols,
        lookback=LOOKBACK,
        forecast_steps=FORECAST_WINDOW,
        lstm_model=model,
        feature_scaler=feat_scaler,
        target_scaler=targ_scaler,
        pressure_model=p_model,
        output_path="forecast.csv",
    )
    print(f"\nSaved forecast CSV: {fcsv}")
    wcsv = generate_wind_csv(df_with_targets, period_hours=0.17, output_path="wind_average.csv")
    print(f"Saved wind CSV:    {wcsv}")


def _full_sequences(df_scaled, feature_cols, lookback):
    arr = df_scaled[feature_cols].values
    return np.array([arr[i - lookback:i] for i in range(lookback, len(arr))],
                    dtype=np.float32)


def _full_seasonality(df_scaled, feature_cols, lookback):
    season_idx = [feature_cols.index(c) for c in SEASONALITY_COLS]
    arr = df_scaled[feature_cols].values
    return np.array([arr[i - 1, season_idx] for i in range(lookback, len(arr))],
                    dtype=np.float32)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
