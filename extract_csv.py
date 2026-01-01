#!/usr/bin/env python3

import redis
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

# Optional imports for advanced formats
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def connect_redis(host='localhost', port=6379, db=0, password=None):
    """Connect to Redis server."""
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=True
    )


def extract_timeseries(r, key, start='-', end='+', count=None):
    """
    Extract time series data from Redis.
    
    Args:
        r: Redis connection
        key: TimeSeries key (e.g., 'sensor:temperature')
        start: Start timestamp ('-' for beginning)
        end: End timestamp ('+' for end)
        count: Optional limit on number of samples
    
    Returns:
        List of (timestamp, value) tuples
    """
    try:
        if count:
            data = r.execute_command('TS.RANGE', key, start, end, 'COUNT', count)
        else:
            data = r.execute_command('TS.RANGE', key, start, end)
        return data
    except redis.exceptions.ResponseError as e:
        print(f"Error extracting data from {key}: {e}")
        return []


def extract_multiple_keys(r, keys, start='-', end='+'):
    """Extract data from multiple time series keys."""
    all_data = {}
    for key in keys:
        data = extract_timeseries(r, key, start, end)
        all_data[key] = data
    return all_data


def timestamp_to_datetime(timestamp_ms):
    """Convert millisecond timestamp to datetime string."""
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')


def save_temperature_data_to_csv(data, output_file, key_name='sensor:temperature'):
    """
    Save time series data to CSV file.
    
    Args:
        data: List of [timestamp, value] pairs
        output_file: Output CSV file path
        key_name: Name of the sensor/key for column naming
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['timestamp_ms', 'datetime', 'value', 'sensor'])
        
        # Write data
        for entry in data:
            timestamp_ms = entry[0]
            value = entry[1]
            dt_str = timestamp_to_datetime(timestamp_ms)
            writer.writerow([timestamp_ms, dt_str, value, key_name])
    
    print(f"Saved {len(data)} records to {output_file}")



def save_humidity_data_to_csv(data, output_file, key_name='sensor:humidity'):
    """
    Save time series data to CSV file.
    
    Args:
        data: List of [timestamp, value] pairs
        output_file: Output CSV file path
        key_name: Name of the sensor/key for column naming
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['timestamp_ms', 'datetime', 'value', 'sensor'])
        
        # Write data
        for entry in data:
            timestamp_ms = entry[0]
            value = entry[1]
            dt_str = timestamp_to_datetime(timestamp_ms)
            writer.writerow([timestamp_ms, dt_str, value, key_name])
    
    print(f"Saved {len(data)} records to {output_file}")


def save_to_json(data, output_file, key_name='sensor:temperature'):
    """Save time series data to JSON file."""
    records = []
    for entry in data:
        records.append({
            'timestamp_ms': entry[0],
            'datetime': timestamp_to_datetime(entry[0]),
            'value': float(entry[1]) if isinstance(entry[1], str) else entry[1],
            'sensor': key_name
        })
    
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Saved {len(data)} records to {output_file}")


def save_to_pandas(data, output_file, key_name='sensor:temperature', format='csv'):
    """
    Save time series data using Pandas (supports CSV, Parquet, HDF5).
    
    Args:
        data: List of [timestamp, value] pairs
        output_file: Output file path
        key_name: Sensor name
        format: 'csv', 'parquet', 'hdf', 'pickle'
    """
    if not PANDAS_AVAILABLE:
        print("Pandas not available. Install with: pip install pandas")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['timestamp_ms', 'value'])
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['sensor'] = key_name
    
    # Set datetime as index for time series analysis
    df.set_index('datetime', inplace=True)
    
    # Save in requested format
    if format == 'csv':
        df.to_csv(output_file)
    elif format == 'parquet':
        df.to_parquet(output_file)
    elif format == 'hdf':
        df.to_hdf(output_file, key='sensor_data', mode='w')
    elif format == 'pickle':
        df.to_pickle(output_file)
    else:
        df.to_csv(output_file)
    
    print(f"Saved {len(df)} records to {output_file} (format: {format})")
    return df


def save_for_ml(data, output_dir, key_name='sensor:temperature', sequence_length=10):
    """
    Save data in format suitable for neural network training.
    Creates sequences for time series prediction.
    
    Args:
        data: List of [timestamp, value] pairs
        output_dir: Output directory
        key_name: Sensor name
        sequence_length: Length of input sequences for LSTM/RNN
    """
    if not NUMPY_AVAILABLE:
        print("NumPy not available. Install with: pip install numpy")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract values
    values = np.array([float(entry[1]) for entry in data])
    timestamps = np.array([entry[0] for entry in data])
    
    # Normalize values (min-max scaling)
    min_val = values.min()
    max_val = values.max()
    values_normalized = (values - min_val) / (max_val - min_val + 1e-8)
    
    # Create sequences for LSTM/RNN
    X, y = [], []
    for i in range(len(values_normalized) - sequence_length):
        X.append(values_normalized[i:i + sequence_length])
        y.append(values_normalized[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Save arrays
    np.save(output_dir / 'X_sequences.npy', X)
    np.save(output_dir / 'y_targets.npy', y)
    np.save(output_dir / 'raw_values.npy', values)
    np.save(output_dir / 'timestamps.npy', timestamps)
    
    # Save normalization parameters
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump({'min': float(min_val), 'max': float(max_val)}, f)
    
    print(f"Saved ML-ready data to {output_dir}")
    print(f"  - X_sequences.npy: shape {X.shape}")
    print(f"  - y_targets.npy: shape {y.shape}")
    print(f"  - raw_values.npy: {len(values)} samples")


def get_all_sensor_keys(r, pattern='sensor:*'):
    """Get all sensor keys matching a pattern."""
    keys = []
    for key in r.scan_iter(match=pattern):
        # Check if it's a time series key
        try:
            r.execute_command('TS.INFO', key)
            keys.append(key)
        except:
            pass
    return keys


def main():
    parser = argparse.ArgumentParser(description='Extract Redis TimeSeries data to CSV/JSON/Parquet')
    parser.add_argument('--host', default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--password', default=None, help='Redis password')
    parser.add_argument('--key', default='sensor:temperature', help='TimeSeries key')
    parser.add_argument('--keys', nargs='+', help='Multiple TimeSeries keys')
    parser.add_argument('--start', default='-', help='Start timestamp (default: beginning)')
    parser.add_argument('--end', default='+', help='End timestamp (default: end)')
    parser.add_argument('--output', default='sensor_data.csv', help='Output file')
    parser.add_argument('--format', choices=['csv', 'json', 'parquet', 'ml'], 
                        default='csv', help='Output format')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length for ML format')
    parser.add_argument('--all-sensors', action='store_true',
                        help='Extract all sensor:* keys')
    
    args = parser.parse_args()
    
    # Connect to Redis
    print(f"Connecting to Redis at {args.host}:{args.port}...")
    r = connect_redis(args.host, args.port, password=args.password)
    
    try:
        r.ping()
        print("Connected successfully!")
    except redis.exceptions.ConnectionError as e:
        print(f"Failed to connect to Redis: {e}")
        return
    
    # Determine keys to extract
    if args.all_sensors:
        keys = get_all_sensor_keys(r)
        print(f"Found {len(keys)} sensor keys: {keys}")
    elif args.keys:
        keys = args.keys
    else:
        keys = [args.key]
    
    # Extract and save data
    for key in keys:
        print(f"\nExtracting data from: {key}")
        data = extract_timeseries(r, key, args.start, args.end)
        
        if not data:
            print(f"No data found for key: {key}")
            continue
        
        print(f"Retrieved {len(data)} data points")
        
        # Generate output filename based on sensor type
        safe_key = key.replace(':', '_')
        output_file = f"{Path(args.output).stem}_{safe_key}{Path(args.output).suffix}"
        
        # Save in requested format
        if args.format == 'csv':
            if 'humidity' in key:
                save_humidity_data_to_csv(data, output_file, key)
            else:
                save_temperature_data_to_csv(data, output_file, key)
        elif args.format == 'json':
            save_to_json(data, output_file, key)
        elif args.format == 'parquet':
            save_to_pandas(data, output_file, key, 'parquet')
        elif args.format == 'ml':
            save_for_ml(data, Path(args.output).stem + '_ml', key, args.sequence_length)


if __name__ == '__main__':
    main()
