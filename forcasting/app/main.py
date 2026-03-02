import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import redis
import os
from datetime import datetime

# Check available devices and GPU usage
print("TensorFlow version:", tf.__version__)
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device.device_type}: {device.name}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU is available: {gpus}")
    # Enable memory growth to avoid allocating all GPU memory at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("GPU is not available. Using CPU.")
def save_csv(start_date, end_date ,target_col='temperature'):
    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date must be provided")
    elif start_date > end_date and start_date != "-" and end_date != "+":
        raise ValueError("start_date must be less than or equal to end_date")
    r = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        # password=os.environ.get("REDIS_PASSWORD"),  # or None if no auth
        db=0,
    )

    ts = r.ts()


    # Fetch each time series separately with timestamps
    temp_data = ts.execute_command("TS.RANGE", "sensor:264041591600404:temperature", start_date, end_date)
    humidity_data = ts.execute_command("TS.RANGE", "sensor:264041591600404:humidity", start_date, end_date)
    pressure_data = ts.execute_command("TS.RANGE", "sensor:264041591600404:pressure", start_date, end_date)

    # Create separate DataFrames for each sensor
    df_temp = pd.DataFrame(temp_data, columns=["timestamp", "temperature"])
    df_humidity = pd.DataFrame(humidity_data, columns=["timestamp", "humidity"])
    df_pressure = pd.DataFrame(pressure_data, columns=["timestamp", "pressure"])

    # Merge on timestamp using outer join to keep all data points
    df = df_temp.merge(df_humidity, on="timestamp", how="outer")
    df = df.merge(df_pressure, on="timestamp", how="outer")

    # Sort by timestamp and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    if target_col not in ['temperature', 'humidity', 'pressure']:
        raise ValueError("Invalid target_col. Must be 'temperature', 'humidity', or 'pressure'.")
    elif target_col == 'temperature':
        df = df[['timestamp', 'temperature']]
    elif target_col == 'humidity':
        df = df[['timestamp', 'humidity']]
    elif target_col == 'pressure':
        df = df[['timestamp', 'pressure']]
    
    return df
    
def forecast_multi_step(model, scaler, df, window_size=10, horizon=24, target_col='temperature'):
    # df already contains the full history you trained on
    df_tmp = df.copy()
    df_tmp['timestamp'] = pd.to_datetime(df_tmp['timestamp'], unit='ms')
    df_tmp.set_index('timestamp', inplace=True)
    target_data = df_tmp[target_col].astype(float).values.reshape(-1, 1)

    # Use the SAME scaler you used in training
    target_scaled = scaler.transform(target_data)

    # Take the last window_size scaled values
    last_window = target_scaled[-window_size:]               # shape (window_size, 1)
    last_window = last_window.reshape((1, window_size, 1))   # (1, window_size, 1)

    future_scaled = []

    window = last_window.copy()
    for _ in range(horizon):
        next_scaled = model.predict(window, verbose=0)       # (1, 1)
        future_scaled.append(next_scaled[0, 0])

        # append prediction and drop oldest step
        next_step = next_scaled.reshape((1, 1, 1))           # (1,1,1)
        window = np.concatenate([window[:, 1:, :], next_step], axis=1)

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_values = scaler.inverse_transform(future_scaled).flatten()

    # build future timestamps (assumes 1 sample == 1 hour)
    last_ts = df_tmp.index[-1]
    future_times = [last_ts + pd.Timedelta(hours=i+1) for i in range(horizon)]

    return future_times, future_values

def load_data(df, window_size=10, target_col='temperature'):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    if target_col not in ['temperature', 'humidity', 'pressure']:
        raise ValueError("Invalid target_col. Must be 'temperature', 'humidity', or 'pressure'.")
    
    target_data = df[target_col].astype(float).values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0,1))
    target_scaled = scaler.fit_transform(target_data)
        
    X,Y = [], []
    for i in range(window_size, len(df)):
        X.append(target_scaled[i - window_size:i, 0])
        Y.append(target_scaled[i, 0])
        
    return np.array(X), np.array(Y), scaler


def train_lstm(X_train, Y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting temperature as an example
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=32)
    return model

def plot_predictions(Y_test, predictions):
    plt.figure(figsize=(12,6))
    plt.plot(Y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Temperature Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.savefig('predictions.png')
# if __name__ == "__main__":
#     # Example usage (timestamps in milliseconds)
#     # now_ms = int(datetime.now().timestamp() * 1000)
#     # start_ms = now_ms - 1000 * 1000  # 1000 seconds ago
#     df = save_csv("-", "+", target_col='temperature')
#     X, Y, scaler = load_data(df, window_size=10, target_col='temperature')
#     # Reshape X for LSTM: (samples, timesteps, features)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#     model = train_lstm(X_train, Y_train)
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1)

#     predictions = model.predict(X_test)
#     predictions = scaler.inverse_transform(predictions).flatten()
#     Y_test = scaler.inverse_transform(Y_test.reshape(-1,1)).flatten()
    
#     rmse = np.sqrt(np.mean((Y_test - predictions)**2))
#     print(f'RMSE: {rmse:.2f}')
        
#     plot_predictions(Y_test, predictions)
    
#     # print(start_ms, now_ms)
    
if __name__ == "__main__":
    df = save_csv("-", "+", target_col='temperature')
    window_size = 10

    X, Y, scaler = load_data(df, window_size=window_size, target_col='temperature')
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = train_lstm(X_train, Y_train)

    # one-step test evaluation (optional)
    preds_test = model.predict(X_test)
    preds_test_inv = scaler.inverse_transform(preds_test).flatten()
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1,1)).flatten()
    rmse = np.sqrt(np.mean((Y_test_inv - preds_test_inv)**2))
    print(f'RMSE: {rmse:.2f}')

    plot_predictions(Y_test_inv, preds_test_inv)

    # ---- 24-hour forecast from end of series ----
    horizon = 24
    future_times, future_temps = forecast_multi_step(
        model=model,
        scaler=scaler,
        df=df,
        window_size=window_size,
        horizon=horizon,
        target_col='temperature'
    )
    time=[]
    values=[]
    for t, v in zip(future_times, future_temps):
        print(t, v)
        time.append(t)
        values.append(v)
        
    plt.figure(figsize=(12,6))
    plt.plot([t.strftime('%Y-%m-%d %H:%M:%S') for t in time], values, marker='o')
    plt.title('24-Hour Temperature Forecast')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('future_forecast.png')