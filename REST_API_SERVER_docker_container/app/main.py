from flask import Flask, request, jsonify
import os
import redis
from datetime import datetime, timezone


def iso_to_unix_ms(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str)  # '2025-12-24T18:41:17'
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # RedisTimeSeries likes ms resolution for IoT; seconds also work
    return int(dt.timestamp() * 1000)

r = redis.Redis(
    host=os.environ.get("REDIS_HOST", "192.168.0.177"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    password=None,  # e.g. REDIS_PASSWORD=ionutqwerty
    db=0,
)

def init_timeseries():
    ts = r.ts()
    # create only if missing
    if not r.exists("sensor:temperature"):
        ts.create("sensor:temperature")
    if not r.exists("sensor:humidity"):
        ts.create("sensor:humidity")

def save_reading(temp_c, humidity_pct, esp_timestamp_str):
    ts = r.ts()
    ts_ms = iso_to_unix_ms(esp_timestamp_str)

    # Add to two separate time series
    ts.add("sensor:temperature", ts_ms, float(temp_c))
    ts.add("sensor:humidity",   ts_ms, float(humidity_pct))

app = Flask(__name__)
@app.route("/sensor", methods=["POST"])
def sensor():
    data = request.get_json(force=True)
    print("Received:", data)
    save_reading(data["temperature"], data["humidity"], data["timestamp"])
    #print("temperature: ", data['temperature'])
    #print("humidity: ", data['humidity'])
    #print("timestamp: ", data['timestamp'])
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=5000,
            ssl_context=("jetson.crt", "jetson.key"))

