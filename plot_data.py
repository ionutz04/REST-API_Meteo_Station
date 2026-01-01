import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Read CSV data
df = pd.read_csv('./sensor_data_sensor_humidity.csv')

# Convert timestamp to datetime for better visualization
df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
sensor_data = np.array(df['value'])

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['datetime'], sensor_data, label='Humidity Sensor Data', color='blue', linewidth=1)

# Format the plot
ax.set_title('Humidity Sensor Data Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Humidity Value (%)', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Format x-axis with readable dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()

# Save to PNG file
fig.savefig('./humidity_sensor_data_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to: ./humidity_sensor_data_plot.png")
plt.close()



