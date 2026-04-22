"""
Production-style runner for the multi-head forecasting pipeline.

Two modes:
  --train      Full training pass on all available Redis data. Saves the
               LSTM model + feature/target scalers + linear pressure model
               under models/ for reuse. Also writes forecast.csv as part of
               the same run. Run this nightly (slow: minutes).
  --predict    Loads the saved artifacts, pulls the latest sensor history
               from Redis, runs ONLY inference, and writes forecast.csv.
               Run this every few minutes (fast: ~2 seconds).

Both modes write the same forecast.csv contract so any downstream consumer
(dashboard, app, alert hook) doesn't need to know which one ran.

Default with no flags: --predict if artifacts exist, otherwise --train.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from predict_rain import (
    REDIS_PATTERN,
    LOOKBACK,
    TEMP_HORIZON,
    FORECAST_WINDOW,
    get_data_from_redis,
    prepare_features,
    make_targets,
    generate_wind_csv,
)
from predict_rain_multihead import (
    add_seasonality,
    apply_feature_scaler,
    PRESSURE_LAG_MINUTES,
    SEASONALITY_COLS,
    generate_forecast_csv_multihead,
    run as run_full_training,
)


MODELS_DIR = Path("models")
ARTIFACTS = {
    "lstm":           MODELS_DIR / "lstm_model.keras",
    "feat_scaler":    MODELS_DIR / "feature_scaler.joblib",
    "target_scaler":  MODELS_DIR / "target_scaler.joblib",
    "pressure_model": MODELS_DIR / "pressure_model.joblib",
    "feature_cols":   MODELS_DIR / "feature_cols.json",
}

FORECAST_CSV = "forecast.csv"
WIND_CSV = "wind_average.csv"


# -------------------------------------------------------------------
# TRAIN MODE  (delegates to predict_rain_multihead.run, then snapshots)
# -------------------------------------------------------------------
def cmd_train() -> None:
    """
    Run the full multi-head training pipeline (LSTM + linear pressure),
    then persist artifacts so cmd_predict() can reload them quickly.

    We use a one-shot trick: monkey-patch the helpers used inside
    predict_rain_multihead.run so we capture the trained objects WITHOUT
    modifying that file. After run() finishes, we save what we captured.
    """
    import predict_rain_multihead as M

    captured = {}

    real_build_multihead_lstm = M.build_multihead_lstm
    real_train_pressure_model = M.train_pressure_model
    real_apply_feature_scaler = M.apply_feature_scaler
    real_apply_target_scaler = M.apply_target_scaler
    real_fit_feature_scaler = M.fit_feature_scaler_on_train
    real_fit_target_scaler = M.fit_target_scaler_on_train

    def spy_fit_feature(df, feature_cols, train_end):
        s = real_fit_feature_scaler(df, feature_cols, train_end)
        captured["feature_scaler"] = s
        captured["feature_cols"] = list(feature_cols)
        return s

    def spy_fit_target(df, target_cols, train_end):
        s = real_fit_target_scaler(df, target_cols, train_end)
        captured["target_scaler"] = s
        captured["target_cols"] = list(target_cols)
        return s

    def spy_build_lstm(n_steps, n_features):
        m = real_build_multihead_lstm(n_steps, n_features)
        captured["lstm"] = m
        return m

    def spy_train_pressure(p_df, train_end, val_end):
        result = real_train_pressure_model(p_df, train_end, val_end)
        captured["pressure_model"] = result
        return result

    M.fit_feature_scaler_on_train = spy_fit_feature
    M.fit_target_scaler_on_train = spy_fit_target
    M.build_multihead_lstm = spy_build_lstm
    M.train_pressure_model = spy_train_pressure
    try:
        run_full_training()
    finally:
        M.fit_feature_scaler_on_train = real_fit_feature_scaler
        M.fit_target_scaler_on_train = real_fit_target_scaler
        M.build_multihead_lstm = real_build_multihead_lstm
        M.train_pressure_model = real_train_pressure_model

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    captured["lstm"].save(ARTIFACTS["lstm"])
    joblib.dump(captured["feature_scaler"], ARTIFACTS["feat_scaler"])
    joblib.dump(captured["target_scaler"], ARTIFACTS["target_scaler"])
    # The pressure_model dict from train_pressure_model has the trained
    # Ridge model + the StandardScaler + feat_cols + (test arrays we don't
    # need at inference, but keeping them is cheap and useful for debug).
    joblib.dump(captured["pressure_model"], ARTIFACTS["pressure_model"])
    with open(ARTIFACTS["feature_cols"], "w") as f:
        json.dump({
            "feature_cols": captured["feature_cols"],
            "target_cols": captured["target_cols"],
            "lookback": LOOKBACK,
            "forecast_window": FORECAST_WINDOW,
            "pressure_lag_minutes": PRESSURE_LAG_MINUTES,
            "seasonality_cols": SEASONALITY_COLS,
        }, f, indent=2)

    print("\n[train] Artifacts saved:")
    for name, path in ARTIFACTS.items():
        print(f"  {name:14s} -> {path}")


# -------------------------------------------------------------------
# PREDICT MODE  (load artifacts, run inference only)
# -------------------------------------------------------------------
def _missing_artifacts() -> list:
    return [str(p) for p in ARTIFACTS.values() if not p.exists()]


def cmd_predict() -> None:
    missing = _missing_artifacts()
    if missing:
        print(f"[predict] Missing artifacts: {missing}", file=sys.stderr)
        print("[predict] Run with --train first.", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    lstm_model = tf.keras.models.load_model(ARTIFACTS["lstm"])
    feat_scaler = joblib.load(ARTIFACTS["feat_scaler"])
    targ_scaler = joblib.load(ARTIFACTS["target_scaler"])
    pressure_model = joblib.load(ARTIFACTS["pressure_model"])
    with open(ARTIFACTS["feature_cols"]) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    print(f"[predict] Loaded artifacts in {time.time()-t0:.2f}s")

    # Pull recent sensor history. We only need enough to cover the LSTM
    # lookback (LOOKBACK 5-min steps = 4h) plus the largest pressure lag
    # (~6h) plus a small safety buffer. get_data_from_redis returns the
    # whole stored series; we slice after prep.
    df = get_data_from_redis()
    if df.empty:
        raise RuntimeError(f"No data loaded from Redis for pattern {REDIS_PATTERN}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = prepare_features(df)
    df = add_seasonality(df)

    # Keep enough history: max(LSTM lookback, max pressure lag) + buffer
    needed_steps = max(LOOKBACK, max(PRESSURE_LAG_MINUTES) // 5) + 20
    if len(df) > needed_steps * 4:
        df = df.tail(needed_steps * 4).reset_index(drop=True)

    # generate_forecast_csv_multihead expects df_original to have the
    # target columns too (used for synth_df rolling pressure context).
    # make_targets drops trailing rows - we don't want that in inference,
    # we want to forecast STARTING FROM the very latest sample. So we
    # add empty target columns directly instead.
    for c in ["temperature_target", "humidity_target", "pressure_target"]:
        if c not in df.columns:
            df[c] = np.nan

    df_scaled = apply_feature_scaler(df, feature_cols, feat_scaler)

    fcsv = generate_forecast_csv_multihead(
        df_original=df,
        df_scaled=df_scaled,
        feature_cols=feature_cols,
        lookback=LOOKBACK,
        forecast_steps=FORECAST_WINDOW,
        lstm_model=lstm_model,
        feature_scaler=feat_scaler,
        target_scaler=targ_scaler,
        pressure_model=pressure_model,
        output_path=FORECAST_CSV,
    )
    wcsv = generate_wind_csv(df, period_hours=0.17, output_path=WIND_CSV)
    print(f"[predict] Wrote {fcsv} and {wcsv} in {time.time()-t0:.2f}s total")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--train", action="store_true",
                   help="Run full training pipeline and snapshot artifacts.")
    g.add_argument("--predict", action="store_true",
                   help="Load saved artifacts and run inference only.")
    args = parser.parse_args()

    try:
        if args.train:
            cmd_train()
        elif args.predict:
            cmd_predict()
        else:
            # Default: predict if artifacts exist, otherwise train.
            if _missing_artifacts():
                print("[run] No artifacts found -> running --train.")
                cmd_train()
            else:
                print("[run] Artifacts found -> running --predict.")
                cmd_predict()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
