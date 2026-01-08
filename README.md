# SHT21 REST API - IoT Temperature & Humidity Monitoring System
Colaborators: Ionescu Ionut, Sandu Laura Florentina
A complete IoT monitoring solution for collecting temperature and humidity data from SHT21 sensors using ESP32 microcontrollers, storing data in Redis TimeSeries, and visualizing with Grafana.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Data Export & Visualization](#data-export--visualization)
- [SNMP Integration](#snmp-integration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

This project provides an end-to-end solution for:
- Reading temperature and humidity from SHT21 sensors via ESP32
- Transmitting sensor data over HTTPS to a REST API server
- Storing time-series data in Redis TimeSeries
- Visualizing data in real-time using Grafana dashboards
- Exporting data to CSV for analysis
- Optional SNMP temperature sensor integration

## 🏗️ Architecture

```
┌─────────────────┐     HTTPS POST      ┌──────────────────┐
│  ESP32 + SHT21  │ ─────────────────►  │  Flask REST API  │
│    (Sensor)     │    /sensor          │   (Port 5000)    │
└─────────────────┘                     └────────┬─────────┘
                                                 │
                                                 ▼
┌─────────────────┐                     ┌──────────────────┐
│     Grafana     │ ◄─────────────────  │      Redis       │
│   (Port 3000)   │    TimeSeries       │   (Port 6379)    │
└─────────────────┘                     └──────────────────┘
                                                 ▲
                                                 │
┌─────────────────┐                              │
│ SNMP Temperature│ ─────────────────────────────┘
│    Sensors      │
└─────────────────┘
```

## 📦 Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| ESP32 Firmware | `esp32_restAPI_implementation/` | PlatformIO project for ESP32 sensor reading |
| REST API Server | `REST_API_SERVER_docker_container/` | Flask API receiving sensor data |
| Redis Database | `redis_docker_container/` | Time-series data storage |
| Grafana | `graphana_docker_container/` | Data visualization dashboard |
| SNMP Interrogator | `snmp_interogator/` | SNMP-based temperature monitoring |
| Data Scripts | Root directory | CSV export and plotting utilities |

## ✅ Prerequisites

- **Docker** and **Docker Compose**
- **Python 3.10+** (for data scripts)
- **PlatformIO** (for ESP32 development)
- **ESP32 Development Board** with SHT21 sensor ( the ESP32 firmware is provided in .hex format)
- **SSL Certificates** (`jetson.crt`, `jetson.key`) for HTTPS

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd sht21_REST_API
```

### 2. Start Redis Container

#### With Docker Compose (for now is not available, because is very buggy)
```bash
cd redis_docker_container
docker-compose up -d
```
#### Or manually:
```bash
cd redis_docker_container
python3 -m venv venv
source venv/bin/activate
pip install redis
python main.py  # Test connection
```

### 3. Start REST API Server

```bash
cd REST_API_SERVER_docker_container

# Ensure SSL certificates are in app/ directory
# cp /path/to/jetson.crt app/
# cp /path/to/jetson.key app/

docker-compose up -d --build
```

### 4. Start Grafana

```bash
cd graphana_docker_container
docker-compose up -d
```
## ⚙️ Configuration

### REST API Server

Environment variables in `REST_API_SERVER_docker_container/docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `your local address of the server (if you run this env locally, not globally)` | Redis server IP address |
| `REDIS_PORT` | `6379` | Redis server port |

### Redis

Environment variables in `redis_docker_container/docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_PASSWORD` | `your_secret_password` | Redis authentication password |

### Grafana

Default credentials:
- **Username:** `admin`
- **Password:** `your_secret_password`
- **URL:** `http://localhost:3000`

### Grafana Data Source Configuration

1. Open Grafana at `http://localhost:3000`
2. Go to **Configuration** → **Data Sources**
3. Add **Redis** data source:
   - Host: `<REDIS_HOST>:6379`
   - Password: `it works right away with no password, but if you set one in redis_docker_container/docker-compose.yml, use it here.`

## � How It Works

This section provides a detailed technical explanation of how each component operates and how data flows through the system.

### 1. Sensor Data Acquisition (ESP32 + SHT21)

The **SHT21** is a digital temperature and humidity sensor that communicates via I²C protocol:

- **Temperature Range:** -40°C to +125°C (±0.3°C accuracy)
- **Humidity Range:** 0% to 100% RH (±2% accuracy)
- **I²C Address:** 0x40

The ESP32 microcontroller:
1. Initializes the I²C bus and connects to the SHT21 sensor
2. Periodically reads temperature and humidity values (configurable interval)
3. Formats the data as JSON with a timestamp
4. Sends an HTTPS POST request to the REST API server

**Data Flow:**
```
SHT21 Sensor → I²C Bus → ESP32 → WiFi → HTTPS POST → Flask Server
```

### 2. REST API Server (Flask)

The Flask application (`REST_API_SERVER_docker_container/app/main.py`) handles incoming sensor data:

**Request Processing:**
1. Receives HTTPS POST request on `/sensor` endpoint
2. Parses JSON payload containing `temperature`, `humidity`, and `timestamp`
3. Generates a UTC timestamp in milliseconds for Redis TimeSeries
4. Stores data in two separate Redis TimeSeries keys

**Code Flow:**
```python
# 1. Request arrives at /sensor endpoint
data = request.get_json()  # {"temperature": 23.5, "humidity": 45.2, "timestamp": "..."}

# 2. Generate current UTC timestamp in milliseconds
ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

# 3. Store in Redis TimeSeries
ts.add("sensor:temperature2", ts_ms, float(temp_c))
ts.add("sensor:humidity2", ts_ms, float(humidity_pct))
```

**Security:**
- HTTPS with SSL/TLS encryption (requires `jetson.crt` and `jetson.key`)
- Runs on port 5000 with host network mode for direct access

**How to create the certificates**
```bash
  openssl genrsa -out ca.key 4096
  openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650   -out ca.pem   -subj "/CN=sht21_demo"
  openssl genrsa -out jetson.key 2048
  openssl req -new -key jetson.key -out jetson.csr   -subj "/CN=192.168.0.177"
  openssl x509 -req -in jetson.csr -CA ca.pem -CAkey ca.key   -CAcreateserial -out jetson.crt -days 365 -sha256
```
>### DISCLAIMER:
>If you generate the certificates using the above commands, the ESP32 will throw an error because it does not trust the self-signed CA. You will need to extract the public key from `ca.pem` and include it in the ESP32 firmware for it to trust the server. In the provided ESP32 firmware, the certificate is the one generated for this demo, is not a generic one.

### 3. Redis TimeSeries Storage

Redis with the **TimeSeries** module provides efficient time-series data storage:

**Data Structure:**
```
Key: sensor:temperature2
     ├── (1704067200000, 23.5)  → timestamp_ms, value
     ├── (1704067260000, 23.7)
     └── (1704067320000, 23.4)

Key: sensor:humidity2
     ├── (1704067200000, 45.2)
     ├── (1704067260000, 46.1)
     └── (1704067320000, 45.8)
```

**Benefits of Redis TimeSeries:**
- **Automatic data compaction:** Reduces storage over time
- **Built-in aggregation:** AVG, MIN, MAX, SUM, COUNT per time bucket
- **Fast range queries:** Efficient retrieval of data within time ranges
- **Memory-efficient:** Optimized for time-series workloads

**Query Examples:**
```bash
# Get last 10 readings
TS.RANGE sensor:temperature2 - + COUNT 10

# Get average per hour for last 24 hours
TS.RANGE sensor:temperature2 - + AGGREGATION avg 3600000
```

### 4. Grafana Visualization

Grafana connects to Redis and displays real-time dashboards:

**Connection Flow:**
```
Grafana → Redis Data Source Plugin → Redis Server → TimeSeries Data
```

**Dashboard Features:**
- **Time-series graphs:** Temperature and humidity over time
- **Gauges:** Current readings with thresholds
- **Alerts:** Notifications when values exceed limits
- **Auto-refresh:** Real-time updates every few seconds

### 5. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [1] SENSOR READING                                                      │
│      SHT21 measures temp=23.5°C, humidity=45.2%                         │
│                           │                                              │
│                           ▼                                              │
│  [2] ESP32 PROCESSING                                                    │
│      Creates JSON: {"temperature": 23.5, "humidity": 45.2,              │
│                     "timestamp": "2025-12-24T18:41:17"}                 │
│                           │                                              │
│                           ▼                                              │
│  [3] HTTPS TRANSMISSION                                                  │
│      POST https://server:5000/sensor                                    │
│      Headers: Content-Type: application/json                            │
│                           │                                              │
│                           ▼                                              │
│  [4] FLASK API PROCESSING                                                │
│      - Validates JSON payload                                           │
│      - Generates UTC timestamp (ms): 1735066877000                      │
│      - Calls Redis TimeSeries API                                       │
│                           │                                              │
│                           ▼                                              │
│  [5] REDIS STORAGE                                                       │
│      TS.ADD sensor:temperature2 1735066877000 23.5                      │
│      TS.ADD sensor:humidity2 1735066877000 45.2                         │
│                           │                                              │
│                           ▼                                              │
│  [6] GRAFANA VISUALIZATION                                               │
│      - Queries Redis every N seconds                                    │
│      - Renders graphs, gauges, and alerts                               │
│      - User views real-time dashboard                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6. SNMP Temperature Integration

The SNMP interrogator (`snmp_interogator/main.py`) provides an alternative data source:

**How SNMP Collection Works:**
1. Queries SNMP-enabled devices using OID (Object Identifier)
2. Parses the response to extract numerical temperature values
3. Stores readings in Redis TimeSeries with sensor-specific keys

**OID Structure:**
```
iso.3.6.1.4.1.17095.5.1.0  →  Enterprise MIB for temperature sensor
│   │ │ │ │ │     │ │ └── Instance (sensor index)
│   │ │ │ │ │     │ └──── Object type (temperature)
│   │ │ │ │ │     └────── Product line
│   │ │ │ │ └──────────── Enterprise ID (17095)
│   │ │ │ └────────────── Private enterprises
│   │ │ └──────────────── Internet
│   │ └────────────────── DOD
│   └──────────────────── ISO
└──────────────────────── Root
```

**Storage Keys:**
```
sensor:snmp_temperature:Hol       → Living room sensor
sensor:snmp_temperature:Mansarda  → Attic sensor
sensor:snmp_temperature:afara     → Outdoor sensor
```

### 7. Data Export Pipeline

The `extract_csv.py` script retrieves data from Redis for offline analysis:

**Process:**
1. Connects to Redis server
2. Queries TimeSeries with `TS.RANGE` command
3. Converts timestamps from milliseconds to ISO format
4. Exports to CSV with columns: `timestamp_ms`, `datetime`, `value`

**Output Files:**
- `sensor_data_sensor_temperature.csv` - All temperature readings
- `sensor_data_sensor_humidity.csv` - All humidity readings
- `sensor_data.csv` - Combined dataset

## 📊 Data Export & Visualization

### Export Data to CSV

```bash
# Install dependencies
pip install redis pandas numpy

# Run the extraction script
python extract_csv.py
```

This creates:
- `sensor_data.csv` - Combined sensor data
- `sensor_data_sensor_temperature.csv` - Temperature readings
- `sensor_data_sensor_humidity.csv` - Humidity readings

### Plot Sensor Data

```bash
# Install dependencies
pip install pandas numpy matplotlib

# Generate plots
python plot_data.py
```

Outputs `humidity_sensor_data_plot.png` with time-series visualization.

## 🌡️ SNMP Integration

The SNMP interrogator collects temperature data from SNMP-enabled devices.

### Configuration

Edit `snmp_interogator/main.py` to configure sensors:

```python
osis = [
    {"name": "Hol", "osi": "iso.3.6.1.4.1.17095.5.1.0"},
    {"name": "Mansarda", "osi": "iso.3.6.1.4.1.17095.5.2.0"},
    {"name": "afara", "osi": "iso.3.6.1.4.1.17095.5.3.0"}
]
```

### Running SNMP Interrogator

```bash
cd snmp_interogator
pip install redis
python main.py
```

## 📡 API Reference

### POST /sensor

Submit sensor reading.

**Request:**
```json
{
  "temperature": 23.5,
  "humidity": 45.2,
  "timestamp": "2025-12-24T18:41:17"
}
```

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl -k -X POST https://localhost:5000/sensor \
  -H "Content-Type: application/json" \
  -d '{"temperature": 23.5, "humidity": 45.2, "timestamp": "2025-12-24T18:41:17"}'
```

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Connection refused to Redis | Verify Redis container is running and check `REDIS_HOST` configuration |
| SSL certificate errors | Ensure `jetson.crt` and `jetson.key` are in the `app/` directory |
| ESP32 not connecting | Check WiFi credentials and server IP in ESP32 firmware |
| Grafana can't connect to Redis | Install Redis data source plugin and verify connection settings |

### Checking Logs

```bash
# REST API Server logs
docker logs sht21_rest_api_server

# Redis logs
docker logs redis_docker_container-redis-1

# Grafana logs
docker logs grafana
```

### Testing Redis Connection

```bash
# From redis_docker_container/
python main.py
```

## 📁 File Structure

```
sht21_REST_API/
├── README.md                           # This file
├── extract_csv.py                      # Export Redis data to CSV
├── plot_data.py                        # Generate data visualizations
├── firmware.hex                       # ESP32 PlatformIO project (firmware provided in .hex format)
│   ├── platformio.ini
│   ├── src/                            # Source code
│   ├── include/
│   ├── lib/
│   └── test/
├── REST_API_SERVER_docker_container/   # Flask REST API
│   ├── docker-compose.yml
│   ├── app/
│   │   └── main.py
│   └── build/
│       └── Dockerfile
├── redis_docker_container/             # Redis TimeSeries
│   ├── docker-compose.yml
│   └── main.py                         # Connection test script
├── graphana_docker_container/          # Grafana visualization
│   └── docker-compose.yml
└── snmp_interogator/                   # SNMP sensor integration
    ├── main.py
    └── osi.txt
```

## 📄 License

This project is provided as-is for educational and personal use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
