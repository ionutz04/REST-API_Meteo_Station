# 🌦️ ESP32 Weather Station - IoT Meteostation Project

---

## 📝 General Description

This project implements a complete **IoT Weather Monitoring Station** built around an ESP32 microcontroller. The system collects environmental data including:

- **Temperature & Humidity** via SHT21 digital sensor (I²C)
- **Wind Speed** via anemometer (pulse counting with interrupt)
- **Wind Direction** via wind vane (analog voltage reading with resistor network)
- **Rainfall** via tipping bucket rain gauge (pulse counting with debounce)

All sensor data is transmitted securely over **HTTPS** to a Flask REST API server, stored in **Redis TimeSeries** database, and visualized in real-time using **Grafana** dashboards.

### Key Features
- Real-time weather data acquisition
- Secure JWT-authenticated data transmission
- Time-series data storage with automatic aggregation
- Web-based visualization dashboards
- CSV data export for offline analysis

---

## 🧾 BOM (Bill Of Materials)

| # | Component | Quantity | Description | Notes |
|---|-----------|----------|-------------|-------|
| 1 | **ESP32 DevKit** | 1 | ESP32 Development Board (e.g., ESP32-WROOM-32) | Main microcontroller |
| 2 | **SHT21** | 1 | Digital Temperature & Humidity Sensor | I²C interface, 0x40 address |
| 3 | **SparkFun Weather Meter Kit** | 1 | [SEN-15901](https://www.sparkfun.com/products/15901) - Includes anemometer, wind vane, and rain gauge | See DS-15901 datasheet |
| 4 | **10kΩ Resistor** | 2 | 1/4W resistors | Voltage divider for wind vane |
| 5 | **Jumper Wires** | ~20 | Male-to-Male and Male-to-Female | Various connections |
| 6 | **Breadboard** | 1 | 830 points (optional) | For prototyping |
| 7 | **5V Power Supply** | 1 | USB or external power source | To power ESP32 |
| 8 | **Enclosure** | 1 | Weatherproof enclosure (optional) | For outdoor deployment |
| 9 | **RJ11 Breakout Boards** | 2 | For connecting weather meter cables | Optional, can splice wires |
| 10 | **Nvidia Jetson Nano Orin 8GB** | 1 | A very powerfull gateway for analisys, forcasting, and AI | Needed for local forcasting and loggin of the data |

### Weather Meter Kit Contents (SEN-15901)
| Sensor | Interface | ESP32 Pin | Notes |
|--------|-----------|-----------|-------|
| Anemometer | Digital (switch) | GPIO32 | Interrupt-based pulse counting |
| Wind Vane | Analog (resistor network) | GPIO33 | ADC with voltage divider |
| Rain Gauge | Digital (tipping bucket switch) | GPIO25 | Debounced interrupt |

## ❓ Project Questions

### Q1 - What is the system boundary?

> *Define what is inside and outside your system. What components are part of your project and what are external dependencies?*

```
The external dependencies are basically the libraries which I used for the ESP32 firmware, and the microservices that I use (e.g., Grafana, Redis, MySQL database). Everything else is built by me, including the `docker-compose.yml` files for these services.
Even the REST API implementation is designed by me, being inspired by the industry implementation of AAA (Authentication, Authorization, and Accounting).
```

---

### Q2 - Where does intelligence live?

> *Where is the decision-making happening? On the edge device (ESP32), in the cloud, or both? What processing happens where?*

```
The intelligence lives on the local server gateway, which in my design is the NVIDIA Jetson Nano Orin. This little board hosts the microservices and will also run the forecasting algorithms used in this project.
```

---

### Q3 - What is the hardest technical problem?

> *Identify the most challenging technical aspect of your project. This could be hardware integration, software complexity, calibration, reliability, etc.*

```
Actually, in this project I have encountered two steps which I consider to be the hardest.
1. The first one is the actual implementation of the `non-deterministic finite automata` logic for the REST API server. At that moment, I could not find a good method for how to separate the cursor connection to the MySQL server, the methods for the `Time Series Data Redis Database`, and the JWT verification steps.
2. The second hardest was actually finding a stable version of the firmware for the ESP32. The initial version caused the ESP32 to constantly disconnect from the server, which was quite annoying because the Docker container running the Flask server kept crashing due to the MySQL cursor freezing the connection. After I made the change in architecture (the automata implementation and the Gunicorn workers implementation), everything worked fine until now.
```

---

### Q4 - What is the minimum demo?

> *Describe the simplest working demonstration of your project that proves the core concept works.*

```
Actually, the simplest working demonstration of this project is to bring the entire meteo station (which is not that big) and manipulate the sensors such that the graphs would change.
```

---

### Q5 - Why is this not just a tutorial?

> *Explain what makes your project unique, what you've added beyond following instructions, or what novel problem you're solving.*

```
This project can be anything but a tutorial, because of a very simple fact: this project is not something you can just copy without understanding or thinking. The entire architecture is actually developed in order to solve a very simple question, but with a very hard background development: **"How can you actually build a meteo station that needs very low latency but very high accuracy in forecasting?"**
```

---

## 🛒 Do You Need an ESP32?

```
YES, of course, because this project is built especially for this kind of scenario, in which you need this kind of system to run continuously without human intervention.
```
