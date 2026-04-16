import numpy as np
import matplotlib.pyplot as plt
import redis 
import pandas as pd

r = redis.Redis(host='192.168.88.168', port=6379, db=0)

def get_data_from_redis():
    ts = r.ts()
    
    df = pd.DataFrame()
    for key in ts.execute_command('KEYS', 'sensor:264041591600404:*'):
        data = ts.range(key, from_time='-', to_time='+')
        if data:
            timestamps, values = zip(*data)
            df[key.decode('utf-8')] = values
    return df

if __name__ == "__main__":
    df = get_data_from_redis()
    print(df.head())
    df.to_csv('data.csv', index=False)
    # Plotting the data
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Sensor Data from Redis Time Series')
    plt.legend()
    plt.show()