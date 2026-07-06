# Rocket AI Controller

Нейросетевая система управления ракетой на ESP32.

## Характеристики

- **Модель:** Полносвязная нейросеть (45,733 параметра)
- **Входы:** 11 параметров (кватернион, угловая скорость, ускорение, угол отклонения)
- **Выходы:** 3 (pitch, yaw, roll)
- **Размер модели:** ~179 КБ (float32)
- **Платформа:** ESP32

##  Структура проекта

```
rocket_ai/
├── sim.py              # Генерация данных
├── train.py            # Обучение модели
├── export.py           # Экспорт в C++ для ESP32
├── rocket_controller.ino # Прошивка для ESP32
├── rocket_model.h      # Веса модели (генерируется)
└── requirements.txt    # Зависимости Python
```

##  Установка

```bash
git clone https://github.com/liquefield-maxouiy/rocket-ai-controller.git
cd rocket-ai-controller
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Использование

1. Генерация данных: `python sim.py`
2. Обучение: `python train.py`
3. Экспорт: `python export.py`
4. Залейте `rocket_controller.ino` на ESP32

## Датчики

- MPU6050 (гироскоп + акселерометр)
- BMP280 (барометр)
