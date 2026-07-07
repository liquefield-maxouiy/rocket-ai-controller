# Rocket AI Controller

Neural network rocket control system for ESP32.

## Specifications

- **Model:** Fully connected neural network (45,733 parameters)
- **Inputs:** 11 parameters (quaternion, angular velocity, acceleration, deflection angle)
- **Outputs:** 3 (pitch, yaw, roll)
- **Model size:** ~179 KB (float32)
- **Platform:** ESP32

## Project Structure

```
rocket_ai/
├── sim.py              # Data generation
├── train.py            # Model training
├── export.py           # Export to C++ for ESP32
├── rocket_controller.ino # Firmware for ESP32
├── rocket_model.h      # Model weights (generated)
└── requirements.txt    # Python dependencies
```

## Installation

```bash
git clone https://github.com/liquefield-maxouiy/rocket-ai-controller.git
cd rocket-ai-controller
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Generate data: `python sim.py`
2. Training: `python train.py`
3. Export: `python export.py`
4. Upload `rocket_controller.ino` to ESP32

## Sensors

- MPU6050 (gyroscope + accelerometer)
- BMP280 (barometer)
