# 🚀 Rocket AI Controller v2.1

> **Neural Network Rocket Control System on ESP32.**  
> Stabilization, autopilot, parachute deployment — all inside the microcontroller.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: ESP32](https://img.shields.io/badge/Platform-ESP32-red)](https://www.espressif.com/)
[![Status: Flight Ready](https://img.shields.io/badge/Status-Flight%20Ready-brightgreen)](https://github.com/liquefield-maxouiy/rocket-ai-controller)

---

## 📖 About the Project

This is an **open-source rocket control system** that uses a **neural network** for real-time flight stabilization.  
The project includes:

- **Data Generation** (`sim.py`) — flight simulator with randomized conditions.
- **Model Training** (`train.py`) — fully-connected neural network (45,733 parameters).
- **Export to C++** (`export.py`) — weights for ESP32.
- **ESP32 Firmware** (`rocket_controller.cpp.ino`) — state machine, sensors, servos, parachute.

---

## 🔥 Key Features

| Feature | Description |
|---------|-------------|
| **Neural Network** | 11 inputs (quat, omega, acc, angle), 3 outputs (pitch, yaw, roll) |
| **State Machine** | 7 states: INIT → GROUND → ARMED → ASCENT → COAST → DESCENT → RECOVERY |
| **Launch Detection** | Acceleration trigger > 2.5G (with false-trigger protection) |
| **Servo Control** | X-configuration (4 fins), speed limiting, smooth startup |
| **Parachute System** | Primary + backup channels, manual test via `p` command |
| **Telemetry** | Serial monitor output: altitude, velocity, state, servo positions |
| **Watchdog** | Hardware timer (3 sec) — crash protection |
| **Calibration** | Automatic MPU6050 and BMP280 calibration on startup |

---

## 🧠 How It Works

### 1. Simulation & Training (on PC)
```bash
python sim.py        # generate data (10,000 episodes)
python train.py      # train neural network (~30 minutes)
python export.py     # export weights to rocket_model.h
```

### 2. ESP32 Firmware Upload

> Load rocket_controller.cpp.ino in Arduino IDE
> Select board: ESP32 Dev Module
> Click "Upload"

### 3. Serial Monitor Commands

| Command | Action |
|---------|--------|
| `a` | Transition from GROUND to ARMED (prepare for launch) |
| `t` | Force transition to ASCENT (test mode) |
| `p` | Manual parachute test (pin 32 → HIGH for 1 second) |

---

## 📦 Hardware Setup

### Wiring Diagram

| Component | ESP32 Pin |
|-----------|-----------|
| MPU6050 (SDA) | GPIO21 |
| MPU6050 (SCL) | GPIO22 |
| BMP280 (SDA)  | GPIO21 |
| BMP280 (SCL)  | GPIO22 |
| SERVO1        | GPIO13 |
| SERVO2        | GPIO14 |
| SERVO3        | GPIO15 |
| SERVO4        | GPIO16 |
| Parachute (signal) | GPIO32 |

### Power Supply

| Device | Power |
|--------|-------|
| ESP32 | USB or 5V from BEC |
| MPU6050, BMP280 | 3.3V from ESP32 |
| MG996R Servos | 5V from BEC (3-5A) |

⚠️ **Important:** Power servos from a separate BEC, NOT from the ESP32!

---

## 🚀 Flight Modes (State Machine)

```
INIT → GROUND → ARMED → ASCENT → COAST → DESCENT → RECOVERY
```

| State | Description |
|-------|-------------|
| **INIT** | Sensor calibration, servo initialization |
| **GROUND** | Waiting for `a` command to transition to ARMED |
| **ARMED** | Monitoring acceleration (waiting for launch > 2.5G) |
| **ASCENT** | Active neural network control, motor burning |
| **COAST** | Motor off, ballistic flight |
| **DESCENT** | Descending, preparing for parachute deployment |
| **RECOVERY** | Parachute deployed, landing |

---

## 🧪 Testing

### 1. Sensor Check
After uploading the firmware, open the serial monitor and verify:
- `MPU6050: initialized and responding`
- `BMP280: found at 0x76`

### 2. Parachute Test
Send command `p` — the LED on pin 32 should light up for 1 second.

### 3. Servo Test
Send command `a`, then `t` — servos should start moving (neural network activates).

---

## 📊 Telemetry Example

```
S:GROUND alt=299.1 vs=0.12 acc=0.99G srv=90,90,90,90 ch1=N ch2=N
S:ARMED alt=299.1 vs=0.15 acc=1.02G srv=90,90,90,90 ch1=N ch2=N
S:ASCENT alt=300.2 vs=2.34 acc=3.12G srv=72,108,82,98 ch1=N ch2=N
S:COAST alt=305.0 vs=5.12 acc=1.01G srv=90,90,90,90 ch1=N ch2=N
S:DESCENT alt=300.0 vs=-3.12 acc=0.99G srv=90,90,90,90 ch1=N ch2=N
S:RECOVERY alt=299.5 vs=-1.12 acc=0.99G srv=90,90,90,90 ch1=Y ch2=N
```

---

## 📄 License

MIT — do whatever you want, but if the rocket crashes, it's on you. 😄

---

## 🔗 Links

- Repository: [github.com/liquefield-maxouiy/rocket-ai-controller](https://github.com/liquefield-maxouiy/rocket-ai-controller)  
- Release v2.1: [github.com/liquefield-maxouiy/rocket-ai-controller/releases/tag/v2.1](https://github.com/liquefield-maxouiy/rocket-ai-controller/releases/tag/v2.1)

---

**🚀 To the stars!**

# **THIS IS A ROCKET MODEL; THE DEVELOPER IS NOT RESPONSIBLE FOR YOUR ACTIONS.**
