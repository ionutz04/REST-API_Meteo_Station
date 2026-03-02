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
    
def load_data(df, window_size=10, target_col='temperature'):
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
if __name__ == "__main__":
    # Example usage (timestamps in milliseconds)
    # now_ms = int(datetime.now().timestamp() * 1000)
    # start_ms = now_ms - 1000 * 1000  # 1000 seconds ago
    df = save_csv("-", "+", target_col='temperature')
    X, Y, scaler = load_data(df, window_size=10, target_col='temperature')
    # Reshape X for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = train_lstm(X_train, Y_train)
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions).flatten()
    Y_test = scaler.inverse_transform(Y_test.reshape(-1,1)).flatten()
    
    rmse = np.sqrt(np.mean((Y_test - predictions)**2))
    print(f'RMSE: {rmse:.2f}')
        
    plot_predictions(Y_test, predictions)
    
    # print(start_ms, now_ms)
    
