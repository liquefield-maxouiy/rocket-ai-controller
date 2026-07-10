# 🚀 NOVA — AI-Powered Rocket Control System

> **Neural network-based rocket control system on ESP32.**  
> Stabilization, autopilot, telemetry, parachute — all inside the microcontroller.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: ESP32](https://img.shields.io/badge/Platform-ESP32-red)](https://www.espressif.com/)
[![Status: Flight Ready](https://img.shields.io/badge/Status-Flight%20Ready-brightgreen)](https://github.com/liquefield-maxouiy/rocket-ai-controller)

---

## 🔥 What Is This?

**NOVA** is a fully autonomous rocket control system that uses a **neural network** for real-time flight stabilization.

The project includes:
- **6-DOF simulator** (`sim.py`) — physically accurate rocket model with 10 types of imperfections.
- **Neural network training** (`train.py`) — 45,733 parameters, trained on 30,000+ episodes.
- **ESP32 flight code** (`rocket_controller.cpp.ino`) — 50 Hz, state machine, sensors, servos, parachute.
- **Web control interface** — Wi-Fi access point, control from your phone without internet.
- **SD card** — telemetry logging in CSV (50 Hz).

---

## 📡 Phone Control

| What | How |
| :--- | :--- |
| **Connect to Wi-Fi** | Network `NOVA_ROCKET`, password `rocket123` |
| **Open website** | `http://192.168.4.1` in your browser |
| **Control** | ARM, LAUNCH, ABORT buttons + real-time telemetry |

---

## 🧠 How It Works

### 1. Simulation & Training (on PC)
```bash
python sim.py        # data generation (10,000 episodes)
python train.py      # neural network training
python export.py     # export weights to rocket_model.h
```

### 2. Flashing ESP32
```bash
# Upload rocket_controller.cpp.ino in Arduino IDE
# Select board: ESP32 Dev Module
# Click "Upload"
```

### 3. Field Launch
- Power on ESP32, BEC, servos.
- Connect to `NOVA_ROCKET` from your phone.
- Open `http://192.168.4.1`.
- Press **ARM**, then **LAUNCH**.

---

## 🔌 Wiring Diagram

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
| SD Card (CS) | GPIO4 |

⚠️ **Servos must be powered from a separate 5V 5A BEC!** ESP32 cannot drive MG995 servos directly.

---

## 📊 State Machine

```
INIT → GROUND → ARMED → ASCENT → COAST → DESCENT → RECOVERY
```

| State | Description |
|-------|-------------|
| **INIT** | Sensor calibration, servo initialization |
| **GROUND** | Waiting for ARM command |
| **ARMED** | Acceleration monitoring (waiting for launch > 2.5G) |
| **ASCENT** | Neural network controls, motor running |
| **COAST** | Motor off, ballistic flight |
| **DESCENT** | Descending, preparing for parachute |
| **RECOVERY** | Parachute deployed, landing |

---

## 🛠️ Hardware

| Component | Model |
|-----------|-------|
| Microcontroller | ESP32 @ 240 MHz |
| IMU | MPU6050 (6-axis, I²C) |
| Barometer | BMP280 (±1 m, I²C) |
| Servos | 4× MG995 (X-config) |
| Parachute | 2× igniter (main + backup) |
| Logging | SD card (CSV @ 50 Hz) |
| Control | Wi-Fi access point |
| Watchdog | Hardware, 3 sec |

---

## 🧪 Testing

| Command (Serial) | Action |
|------------------|--------|
| `a` | Transition GROUND → ARMED |
| `t` | Force transition to ASCENT (test) |
| `p` | Manual parachute test (pin 32) |
| `d` | Force transition from COAST to DESCENT |

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

## 📁 Project Structure

```
rocket-ai-controller/
├── sim.py                    # 6-DOF simulator
├── train.py                  # Neural network training
├── export.py                 # Export to C++
├── rocket_controller.cpp.ino # ESP32 firmware
├── rocket_model.h            # Model weights
├── index.html                # Web control interface
└── README.md                 # You are here
```

---

## 📄 License

MIT — do whatever you want, but if the rocket crashes, it's your own fault.

---

## 🔗 Links

- Repository: [github.com/liquefield-maxouiy/rocket-ai-controller](https://github.com/liquefield-maxouiy/rocket-ai-controller)
- Releases: [github.com/liquefield-maxouiy/rocket-ai-controller/releases](https://github.com/liquefield-maxouiy/rocket-ai-controller/releases)
- Project website: [liquefield-maxouiy.github.io/rocket-ai-controller](https://liquefield-maxouiy.github.io/rocket-ai-controller)

---

**🚀 Onward, to the stars!**

---

# **THIS IS A ROCKET MODEL; THE DEVELOPER IS NOT RESPONSIBLE FOR YOUR ACTIONS.**
