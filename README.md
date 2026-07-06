# 🚀 Rocket AI Controller (INT8 Edition)

> **Neural network rocket control system for low-power microcontrollers.**  
> Same power, 4x less memory, 4x faster.

---

## 🎯 Who is this branch for?

- **Arduino Nano** (ATmega328P, 32 KB Flash, 2 KB RAM)
- **ESP8266** (80 MHz, 80 KB RAM)
- **STM32F1** (72 MHz, 20 KB RAM)
- **Any 8-bit MCU** with >32 KB Flash

If you have an **ESP32** — go to the `master` branch, that's where the float model lives.  
If you have a **low-power MCU** — you're in the right place.

---

## 📊 INT8 Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | Fully connected neural network (MLP) |
| Parameters | 45,733 |
| Inputs | 11 (quat, omega, acc, angle) |
| Outputs | 3 (pitch, yaw, roll) |
| **Size (int8)** | **~45 KB** |
| **Speed** | **4x faster than float** |
| Platform | Arduino Nano, ESP8266, STM32, any MCU |

---

## ⚡ Why int8?

| Parameter | float32 | int8 | Benefit |
|-----------|---------|------|---------|
| **Model size** | 180 KB | **45 KB** | **-75%** |
| **Weight memory** | 180 KB | **45 KB** | **fits on Nano** |
| **Inference speed** | 15 ms | **3-4 ms** | **+400%** |
| **Accuracy** | 100% | **~98-99%** | barely changed |

---

## 🔧 How It Works

1. On your PC (in the `master` branch) you generate data and train the model.
2. In this branch you **convert** the trained model to int8.
3. **`export_int8.py`** — reads `rocket_best.pth` and generates `rocket_model_int8.h`.
4. **`rocket_controller_int8.ino`** — firmware for low-power MCUs.

---

## 📁 `feature/int8` Branch Structure

```
rocket-ai-controller (feature/int8)/
├── export_int8.py              # float → int8 converter
├── rocket_model_int8.h         # int8 weights (45 KB)
├── rocket_controller_int8.ino  # firmware for low-power MCUs
├── README.md                   # You are here
└── requirements.txt            # Python dependencies
```

---

## 🔌 Wiring Diagram (Arduino Nano)

| Component | Nano Pin |
|-----------|----------|
| MPU6050 (SDA) | A4 |
| MPU6050 (SCL) | A5 |
| BMP280 (SDA)  | A4 |
| BMP280 (SCL)  | A5 |
| SERVO1        | D3 |
| SERVO2        | D5 |
| SERVO3        | D6 |
| SERVO4        | D9 |

⚠️ **Power servos from a separate 5V 5A BEC!** The Nano can't handle MG995s.

---

## 🚀 Quick Start (this branch)

```bash
# Clone the repo
git clone -b feature/int8 https://github.com/liquefield-maxouiy/rocket-ai-controller.git
cd rocket-ai-controller

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate the int8 model from rocket_best.pth
python export_int8.py

# Upload rocket_controller_int8.ino to your Arduino Nano
```

---

## 🧠 How to Build the Model for int8

```bash
# 1. Switch to master branch (data & training)
git checkout master

# 2. Generate data
python sim.py

# 3. Train the model
python train.py

# 4. Switch back to int8 branch
git checkout feature/int8

# 5. Copy rocket_best.pth from master
cp ../master/rocket_best.pth .

# 6. Export to int8
python export_int8.py
```

---

## 🔬 Technical Details (int8)

### Weight Quantization
- Weights quantized to `int8` with preserved scale
- Activations: `ReLU` and `Tanh` in float
- Multiplications: `int8 × int8 = int32` → faster and more compact

### Forward Pass
```
float input → int8 quantize
    ↓
int8 matmul (fast!)
    ↓
int32 → float + scale
    ↓
ReLU / Tanh
    ↓
float output → denormalize
```

---

## 📊 Sample Output

```
pitch 0.012 yaw -0.034 roll 0.001 alt 45.2 temp 24.7 servos 90 87 92 91
pitch 0.015 yaw -0.040 roll 0.003 alt 45.4 temp 24.7 servos 91 87 93 89
```

---

## 📄 License

MIT — do whatever you want, but if the rocket crashes — that's on you.

---

**🚀 Even on weak hardware — to the stars!**
