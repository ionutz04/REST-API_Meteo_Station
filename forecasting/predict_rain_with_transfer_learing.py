"""
Transfer-learning variant of predict_rain.py.

Flow:
  1. PRETRAIN the same multi-output LSTM (temperature/humidity/pressure
     regression) used in predict_rain.py on a Ploiești-region historical
     dataset built from otopeni_pws_2025_metric.csv.
       * If the Otopeni CSV has T/H/P, it is used directly.
       * Otherwise we fetch Open-Meteo's Ploiești archive (free, no key)
         for the same year so the regression head has real data, and we
         merge the high-resolution Otopeni rainfall back in.
  2. FINE-TUNE on the local Redis sensor data.
  3. Produce the same forecast.csv / wind_average.csv / debug_plots as
     predict_rain.py, using the pressure-drop heuristic for rain
     probability (consistent with predict_rain.py).

Caching:
  * pretrain_lstm_weights.h5 -> if present, pretraining is skipped.
    Set PRETRAIN_FORCE=1 to retrain.
"""

from __future__ import annotations

import os
import sys
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from predict_rain import (
    REDIS_PATTERN,
    LOOKBACK,
    TEMP_HORIZON,
    FORECAST_WINDOW,
    TRAIN_RATIO,
    VAL_RATIO,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COLS,
    TARGET_OUT_NAMES,
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
    build_multi_output_model,
    evaluate_temperature,
    generate_forecast_csv,
    generate_wind_csv,
    plot_temp_training,
    plot_temp_mae,
    plot_temp_forecast,
    plot_predicted_data_over_entire_series,
)


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PRETRAIN_CSV_PATH = os.environ.get(
    "PRETRAIN_CSV_PATH",
    "otopeni_pws_2025_metric.csv",
)
OPEN_METEO_CACHE = "otopeni_pws_2025_metric.openmeteo.csv"
PRETRAIN_WEIGHTS_PATH = "pretrain_lstm.weights.h5"

PLOIESTI_LAT = 44.9333
PLOIESTI_LON = 26.0167
OPEN_METEO_YEARS = 5

# Use only the last N days of the pretraining dataset so the prior
# reflects the *current* season instead of an annual mean. Set to 0 to
# disable the filter and use the whole dataset.
PRETRAIN_RECENT_DAYS = 21

PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH_SIZE = 128
PRETRAIN_LR = 1e-3

FINETUNE_EPOCHS = 80
FINETUNE_BATCH_SIZE = 32
FINETUNE_LR = 5e-4

# Weight the most-recent slice of the fine-tune training set higher so
# the model adapts to the current weather regime instead of averaging
# over the older, possibly different, conditions.
FINETUNE_RECENT_FRACTION = 0.25
FINETUNE_RECENT_WEIGHT = 3.0

FREEZE_TRUNK_DURING_FINETUNE = False

# Cadence the local Redis sensor produces. We resample the pretraining
# dataset to this cadence so the LSTM learns dynamics at the same time
# scale used during inference (otherwise an LSTM trained on hourly steps
# rolled out at 5-min steps fakes a whole diurnal cycle in a few hours).
INFERENCE_CADENCE_MINUTES = 5

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# -------------------------------------------------------------------
# DATASET LOADING + NORMALIZATION
# -------------------------------------------------------------------
COLUMN_ALIASES = {
    "temperature": ["temperature", "temperature_c", "temp", "temp_c",
                    "temperature_2m", "t2m"],
    "humidity": ["humidity", "humidity_pct", "rh", "relative_humidity",
                 "relative_humidity_2m"],
    "pressure": ["pressure", "pressure_hpa", "pressure_msl",
                 "surface_pressure", "slp"],
    "wind_speed": ["wind_speed", "wind_speed_ms", "windspeed",
                   "wind_speed_10m"],
    "wind_direction_degrees": ["wind_direction_degrees", "wind_dir_deg",
                               "wind_direction", "wind_direction_10m"],
    "rainfall": ["rainfall", "precip_rate_mm_h", "precipitation",
                 "rain", "precip"],
    "timestamp": ["timestamp", "obs_time_utc", "time", "datetime",
                  "obs_time"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lowered = {c.lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a.lower() in lowered:
                rename[lowered[a.lower()]] = canonical
                break
    return df.rename(columns=rename)


def _to_naive_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_convert("UTC").dt.tz_localize(None)


def _download_open_meteo(path: str, years: int = OPEN_METEO_YEARS) -> str:
    end = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=2)
    start = end - pd.DateOffset(years=years)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={PLOIESTI_LAT}&longitude={PLOIESTI_LON}"
        f"&start_date={start.date().isoformat()}"
        f"&end_date={end.date().isoformat()}"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
        "wind_speed_10m,wind_direction_10m,precipitation"
        "&timezone=UTC"
    )
    print(f"[pretrain] Downloading Open-Meteo archive...\n[pretrain]   {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    h = payload["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "temperature": h.get("temperature_2m"),
        "humidity": h.get("relative_humidity_2m"),
        "pressure": h.get("pressure_msl"),
        "wind_speed": h.get("wind_speed_10m"),
        "wind_direction_degrees": h.get("wind_direction_10m"),
        "rainfall": h.get("precipitation"),
    })
    df.to_csv(path, index=False)
    print(f"[pretrain] Cached Open-Meteo archive to {path} ({len(df)} rows).")
    return path


def _load_csv_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} has no recognizable timestamp column.")
    df["timestamp"] = _to_naive_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in ["temperature", "humidity", "pressure",
                "wind_speed", "wind_direction_degrees", "rainfall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan if col in ("temperature", "humidity", "pressure") else 0.0
    return df


def load_pretrain_dataframe() -> pd.DataFrame:
    """
    Build the pretraining dataframe from the Otopeni CSV. If Otopeni has no
    temperature/humidity/pressure (the raw export only contains rainfall),
    we use Open-Meteo for T/H/P and overlay the Otopeni rainfall by hour.
    """
    if not Path(PRETRAIN_CSV_PATH).exists():
        raise FileNotFoundError(
            f"Pretraining CSV {PRETRAIN_CSV_PATH} is missing. "
            "Place it in the working directory or set PRETRAIN_CSV_PATH."
        )
    otopeni = _load_csv_clean(PRETRAIN_CSV_PATH)
    print(f"[pretrain] Otopeni rows loaded: {len(otopeni)}")

    has_thp = otopeni[["temperature", "humidity", "pressure"]].notna().sum().sum() > 0

    if has_thp:
        df = otopeni
        # Drop rows where ALL of temperature/humidity/pressure are missing.
        df = df.loc[df[["temperature", "humidity", "pressure"]].notna().any(axis=1)]
        df = df.reset_index(drop=True)
        print(f"[pretrain] Using Otopeni T/H/P directly ({len(df)} usable rows).")
    else:
        print("[pretrain] Otopeni CSV has no T/H/P columns -> "
              "supplementing with Open-Meteo archive.")
        if not Path(OPEN_METEO_CACHE).exists():
            _download_open_meteo(OPEN_METEO_CACHE)
        df = _load_csv_clean(OPEN_METEO_CACHE)
        # Overlay Otopeni rainfall (5-min) into Open-Meteo (hourly) by max-per-hour.
        otop_rain = otopeni[["timestamp", "rainfall"]].dropna()
        if len(otop_rain) > 0:
            otop_rain = otop_rain.copy()
            otop_rain["timestamp"] = otop_rain["timestamp"].dt.floor("h")
            hourly = otop_rain.groupby("timestamp", as_index=False)["rainfall"].max()
            df = df.merge(hourly, on="timestamp", how="left", suffixes=("", "_otopeni"))
            df["rainfall"] = df["rainfall_otopeni"].combine_first(df["rainfall"])
            df = df.drop(columns=["rainfall_otopeni"])
            print(f"[pretrain] Overlaid {len(hourly)} Otopeni rainfall hours.")

    # Final sanity: physical clipping.
    df["temperature"] = df["temperature"].clip(-50.0, 60.0)
    df["humidity"] = df["humidity"].clip(0.0, 100.0)
    df["pressure"] = df["pressure"].clip(870.0, 1085.0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0.0, upper=80.0)
    df["wind_direction_degrees"] = df["wind_direction_degrees"].clip(0.0, 360.0)
    df["rainfall"] = df["rainfall"].clip(lower=0.0, upper=500.0)

    # ---- Keep only the most recent N days so the pretrained prior
    # matches the current season instead of an annual average. ----
    if PRETRAIN_RECENT_DAYS and len(df) > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=PRETRAIN_RECENT_DAYS)
        before = len(df)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        print(f"[pretrain] Seasonal slice last {PRETRAIN_RECENT_DAYS}d: "
              f"{before} -> {len(df)} rows")

    print(f"[pretrain] Final pretraining frame: {len(df)} rows  "
          f"range {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # ---- Resample to local-sensor cadence so the LSTM learns dynamics
    # at the same time scale used during inference. ----
    df = _resample_to_cadence(df, minutes=INFERENCE_CADENCE_MINUTES)
    print(f"[pretrain] Resampled to {INFERENCE_CADENCE_MINUTES}-min cadence: "
          f"{len(df)} rows.")
    return df


# -------------------------------------------------------------------
# SEASONALITY FEATURES (cheap day-of-year + hour-of-day priors)
# -------------------------------------------------------------------
SEASONALITY_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def _add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append cyclical hour-of-day and day-of-year encodings. These give the
    model a cheap seasonal prior without needing historical analog lookups.
    """
    if "timestamp" not in df.columns:
        raise KeyError("seasonality requires a 'timestamp' column")
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear + (ts.dt.hour / 24.0)
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def _seasonality_row(ts: pd.Timestamp) -> dict:
    hour = ts.hour + ts.minute / 60.0
    doy = ts.dayofyear + (ts.hour / 24.0)
    return {
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
    }


# -------------------------------------------------------------------
# FORECAST (seasonality-aware recursive roll-out)
# -------------------------------------------------------------------
def generate_forecast_csv_with_seasonality(
    df_scaled,
    df_original,
    feature_cols,
    target_cols,
    lookback,
    forecast_steps,
    model,
    feature_scaler,
    target_scaler,
    output_path="forecast.csv",
):
    """
    Mirrors predict_rain.generate_forecast_csv but updates the
    hour_sin/hour_cos/doy_sin/doy_cos columns for each future step so the
    model sees the correct time-of-day / day-of-year at each horizon.
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

    hist_pressure = pd.to_numeric(df_original["pressure"], errors="coerce").dropna()
    hist_humidity = pd.to_numeric(df_original["humidity"], errors="coerce").dropna()
    hist_temperature = pd.to_numeric(df_original["temperature"], errors="coerce").dropna()
    hist_timestamps = pd.to_datetime(df_original["timestamp"]).reset_index(drop=True)

    target_to_feature_idx = {t: feature_cols.index(t) for t in target_cols if t in feature_cols}
    seasonal_idx = {c: feature_cols.index(c) for c in SEASONALITY_COLS if c in feature_cols}

    pred_temps, pred_hums, pred_press = [], [], []
    pred_timestamps = []
    rows = []

    for step in range(1, forecast_steps + 1):
        inp = window[np.newaxis, :, :]
        preds = model.predict(inp, verbose=0)[0]
        if target_scaler is not None:
            preds = target_scaler.inverse_transform(preds.reshape(1, -1))[0]
        pred_map = {name: float(val) for name, val in zip(target_cols, preds)}

        forecast_ts = last_ts + bucket_td * step
        pred_timestamps.append(forecast_ts)
        pred_temps.append(pred_map.get("temperature", np.nan))
        pred_hums.append(pred_map.get("humidity", np.nan))
        pred_press.append(pred_map.get("pressure", np.nan))

        synth_df = pd.DataFrame({
            "timestamp": pd.concat(
                [hist_timestamps, pd.Series(pred_timestamps)], ignore_index=True),
            "pressure": pd.concat(
                [hist_pressure, pd.Series(pred_press)], ignore_index=True),
            "humidity": pd.concat(
                [hist_humidity, pd.Series(pred_hums)], ignore_index=True),
            "temperature": pd.concat(
                [hist_temperature, pd.Series(pred_temps)], ignore_index=True),
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

        # Build the next raw feature row:
        #   - target columns come from the model prediction,
        #   - seasonality columns are recomputed for the *next* timestamp,
        #   - any other feature stays at its last observed value.
        new_raw = last_raw.copy()
        for tgt, feat_idx in target_to_feature_idx.items():
            new_raw[feat_idx] = pred_map[tgt]
        next_ts = forecast_ts + bucket_td  # features describe the NEXT step's inputs
        season = _seasonality_row(next_ts)
        for col, feat_idx in seasonal_idx.items():
            new_raw[feat_idx] = season[col]

        new_scaled = feature_scaler.transform(
            pd.DataFrame([new_raw], columns=feature_cols)
        ).flatten()
        window = np.vstack([window[1:], new_scaled])
        last_raw = new_raw

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    return str(output_path)


def _resample_to_cadence(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Resample an irregular / coarser timeseries to a uniform cadence.
      - T/H/P/wind: linear interpolation between observed samples
        (with reasonable bounded extrapolation).
      - rainfall: spread the original bucket's total uniformly across
        the higher-cadence sub-buckets so the rain horizon labels stay
        consistent.
    """
    if len(df) == 0:
        return df
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    new_index = pd.date_range(df.index.min(), df.index.max(),
                              freq=f"{minutes}min")

    cont_cols = ["temperature", "humidity", "pressure",
                 "wind_speed", "wind_direction_degrees"]
    cont_cols = [c for c in cont_cols if c in df.columns]
    cont = df[cont_cols].reindex(
        df.index.union(new_index)
    ).interpolate(method="time").reindex(new_index)

    out = cont.copy()

    if "rainfall" in df.columns:
        rain = df["rainfall"].reindex(
            df.index.union(new_index)
        ).reindex(new_index)
        # Backfill so each new sub-bucket inherits the next observed
        # rainfall total, then divide by the number of sub-buckets per
        # original bucket. For Open-Meteo hourly that's 60/5 = 12.
        sub = max(1, int(round(60 / minutes)))
        rain = rain.bfill(limit=sub).fillna(0.0) / sub
        out["rainfall"] = rain.values

    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out


# -------------------------------------------------------------------
# PRETRAIN
# -------------------------------------------------------------------
def pretrain_model(feature_cols, target_out_names) -> tf.keras.Model:
    raw = load_pretrain_dataframe()

    df = prepare_features(raw)
    df = _add_seasonality(df)
    df = make_targets(df, horizon=TEMP_HORIZON)
    print(f"[pretrain] After feature/target prep: {len(df)} rows")

    train_end_row, val_end_row = chronological_split_indices(
        len(df), TRAIN_RATIO, VAL_RATIO,
    )

    feature_scaler = fit_feature_scaler_on_train(df, feature_cols, train_end_row)
    df_scaled = apply_feature_scaler(df, feature_cols, feature_scaler)

    target_scaler = fit_target_scaler_on_train(df_scaled, target_out_names, train_end_row)
    df_scaled = apply_target_scaler(df_scaled, target_out_names, target_scaler)

    X_tr, Y_tr, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, 0, train_end_row,
    )
    X_va, Y_va, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, train_end_row, val_end_row,
    )
    print(f"[pretrain] sequences  train={X_tr.shape}  val={X_va.shape}")

    if len(X_tr) == 0:
        raise RuntimeError("Pretraining dataset produced no training sequences.")

    model = build_multi_output_model(
        n_steps=X_tr.shape[1], n_features=X_tr.shape[2],
        n_targets=Y_tr.shape[1],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PRETRAIN_LR),
        loss="mse",
        metrics=["mae"],
    )

    early_stop = EarlyStopping(
        monitor="val_mae", patience=6, restore_best_weights=True,
        mode="min", verbose=1,
    )
    model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=PRETRAIN_EPOCHS,
        batch_size=PRETRAIN_BATCH_SIZE,
        callbacks=[early_stop],
        shuffle=False,
        verbose=1,
    )

    model.save_weights(PRETRAIN_WEIGHTS_PATH)
    print(f"[pretrain] Saved weights to {PRETRAIN_WEIGHTS_PATH}")
    return model


# -------------------------------------------------------------------
# FINE-TUNE + FORECAST (mirrors predict_rain.py main flow)
# -------------------------------------------------------------------
def _freeze_trunk(model: tf.keras.Model) -> None:
    if not FREEZE_TRUNK_DURING_FINETUNE:
        return
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            layer.trainable = False
            print(f"[finetune] Froze layer: {layer.name}")
            break


def finetune_and_run() -> None:
    feature_cols = ["temperature", "humidity", "pressure"] + SEASONALITY_COLS
    target_cols = TARGET_COLS
    target_out_names = TARGET_OUT_NAMES

    # ---- Stage 1: pretraining (or skip if cached weights exist) ----
    force = os.environ.get("PRETRAIN_FORCE", "0") == "1"
    pretrained_model = None
    if Path(PRETRAIN_WEIGHTS_PATH).exists() and not force:
        print(f"[pretrain] Found {PRETRAIN_WEIGHTS_PATH} -> skipping pretraining.")
        print("[pretrain] (set PRETRAIN_FORCE=1 to retrain)")
    else:
        if force and Path(PRETRAIN_WEIGHTS_PATH).exists():
            print("[pretrain] PRETRAIN_FORCE=1 -> retraining.")
        else:
            print("[pretrain] No cached weights -> running pretraining stage.")
        pretrained_model = pretrain_model(feature_cols, target_out_names)

    # ---- Stage 2: load Redis local sensor data ----
    df = get_data_from_redis()
    if df.empty:
        raise RuntimeError(f"No data loaded from Redis for pattern {REDIS_PATTERN}")
    if "timestamp" not in df.columns:
        raise KeyError("Missing timestamp column after Redis merge.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = prepare_features(df)
    df = _add_seasonality(df)
    df = make_targets(df, horizon=TEMP_HORIZON)
    print(f"\n[finetune] Local rows after prep: {len(df)}")

    current_pressure_rain_prob = estimate_rain_probability_from_pressure(df)
    print(f"[finetune] Pressure-based rain probability (current): "
          f"{current_pressure_rain_prob:.3f}")

    train_end_row, val_end_row = chronological_split_indices(
        len(df), TRAIN_RATIO, VAL_RATIO,
    )

    feature_scaler = fit_feature_scaler_on_train(df, feature_cols, train_end_row)
    df_scaled = apply_feature_scaler(df, feature_cols, feature_scaler)

    target_scaler = fit_target_scaler_on_train(df_scaled, target_out_names, train_end_row)
    df_scaled = apply_target_scaler(df_scaled, target_out_names, target_scaler)

    X_tr, Y_tr, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, 0, train_end_row,
    )
    X_va, Y_va, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, train_end_row, val_end_row,
    )
    X_te, Y_te, _ = build_multi_target_sequences(
        df_scaled, feature_cols, target_out_names, LOOKBACK, val_end_row, len(df_scaled),
    )
    print("[finetune] sequence shapes:")
    print("  train:", X_tr.shape, Y_tr.shape)
    print("  val:  ", X_va.shape, Y_va.shape)
    print("  test: ", X_te.shape, Y_te.shape)

    # ---- Stage 3: build fine-tune model + transfer weights ----
    model = build_multi_output_model(
        n_steps=X_tr.shape[1], n_features=X_tr.shape[2],
        n_targets=Y_tr.shape[1],
    )

    if pretrained_model is not None:
        model.set_weights(pretrained_model.get_weights())
        print("[finetune] Transferred weights from in-memory pretrained model.")
    elif Path(PRETRAIN_WEIGHTS_PATH).exists():
        model.load_weights(PRETRAIN_WEIGHTS_PATH)
        print(f"[finetune] Loaded pretrained weights from {PRETRAIN_WEIGHTS_PATH}.")
    else:
        print("[finetune] WARNING: no pretrained weights; training from scratch.")

    _freeze_trunk(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="mse",
        metrics=["mae"],
    )

    early_stop = EarlyStopping(
        monitor="val_mae", patience=8, restore_best_weights=True,
        mode="min", verbose=1,
    )

    # Recency-weighted fine-tuning: emphasise the most-recent slice of
    # the training set so the model adapts to current conditions.
    sw = np.ones(len(X_tr), dtype=np.float32)
    if FINETUNE_RECENT_FRACTION > 0 and len(sw) > 0:
        n_recent = max(1, int(round(len(sw) * FINETUNE_RECENT_FRACTION)))
        sw[-n_recent:] = FINETUNE_RECENT_WEIGHT
        print(f"[finetune] sample_weight: last {n_recent}/{len(sw)} samples "
              f"@ x{FINETUNE_RECENT_WEIGHT}")

    history = model.fit(
        X_tr, Y_tr,
        sample_weight=sw,
        validation_data=(X_va, Y_va),
        epochs=FINETUNE_EPOCHS,
        batch_size=FINETUNE_BATCH_SIZE,
        callbacks=[early_stop],
        shuffle=False,
        verbose=1,
    )

    # ---- Stage 4: evaluate + plot + write CSVs (same as predict_rain.py) ----
    Y_pred_test_scaled = model.predict(X_te, verbose=0)
    Y_test_real = target_scaler.inverse_transform(Y_te)
    Y_pred_test = target_scaler.inverse_transform(Y_pred_test_scaled)

    print("\n[finetune] Per-target test metrics (real units):")
    for i, name in enumerate(target_cols):
        mae, rmse = evaluate_temperature(Y_test_real[:, i], Y_pred_test[:, i])
        print(f"  {name:12s}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    print("\n[finetune] Latest one-step predictions (real units):")
    latest_window = X_te[-1:].copy() if len(X_te) > 0 else X_va[-1:].copy()
    latest_pred_scaled = model.predict(latest_window, verbose=0)
    latest_pred = target_scaler.inverse_transform(latest_pred_scaled)[0]
    for name, val in zip(target_cols, latest_pred):
        print(f"  {name}: {val:.3f}")

    loss_plot = plot_temp_training(history, output_dir=OUTPUT_DIR)
    mae_plot = plot_temp_mae(history, output_dir=OUTPUT_DIR)
    forecast_plots = [
        plot_temp_forecast(Y_test_real[:, i], Y_pred_test[:, i],
                           target_cols[i], output_dir=OUTPUT_DIR)
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

    forecast_csv = generate_forecast_csv_with_seasonality(
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


if __name__ == "__main__":
    try:
        finetune_and_run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
