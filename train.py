import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import glob
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Фиксируем сиды
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Ускорение матричных умножений на CPU
torch.set_float32_matmul_precision('high')
torch.set_num_threads(24) #Yadra xeon

print("=" * 60)
print("Обучение ИИ для ESP32")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"Seed: {SEED}")
print("=" * 60)

#Проверка данных
files = glob.glob("flight_data_*.log.gz")
if not files:
    print("Файлы не найдены!")
    exit()

log_file = sorted(files)[-1]
print(f" Файл: {log_file}")
print(f" Размер: {os.path.getsize(log_file) / 1024 ** 2:.1f} МБ")

# Загрузка чанками (экономит RAM)
print("\n Загрузка данных чанками...")
start_load = time.time()

features = [
    'quat_x', 'quat_y', 'quat_z', 'quat_w',
    'omega_x', 'omega_y', 'omega_z',
    'acc_x', 'acc_y', 'acc_z',
    'angle_deg'
]

targets = ['action_pitch', 'action_yaw', 'action_roll']

all_X = []
all_y = []
total_rows = 0

# Читаем по 100K строк за раз
for chunk in pd.read_csv(log_file, compression='gzip', chunksize=100000, low_memory=False):
    # Фильтрация: убираем нереалистичные состояния
    chunk = chunk[chunk['alt'] > 0.5]
    chunk = chunk[chunk['thrust'] > 50]
    chunk = chunk[np.abs(chunk['angle_deg']) < 60]

    # Убираем выбросы внутри чанка
    for col in features + targets:
        mean = chunk[col].mean()
        std = chunk[col].std()
        if std > 0:
            chunk = chunk[np.abs(chunk[col] - mean) < 4 * std]

    if len(chunk) > 0:
        all_X.append(chunk[features].values.astype(np.float32))
        all_y.append(chunk[targets].values.astype(np.float32))
        total_rows += len(chunk)

    print(f"Загружено {total_rows:,} строк...", end='\r')

# Объединяем все чанки
X = np.vstack(all_X)
y = np.vstack(all_y)
del all_X, all_y  # Чистим память

print(f"\n Загружено {total_rows:,} строк за {time.time() - start_load:.1f}с")
print(f"   X shape: {X.shape}, y shape: {y.shape}")
print(f"   Память: {X.nbytes / 1024 ** 2:.1f} + {y.nbytes / 1024 ** 2:.1f} МБ")

# Нормализация
print("\nНормализация...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Разделение (только если данных много)
print("\nРазделение данных...")

if total_rows > 500000:
    # Для больших данных — 80/10/10
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )
else:
    # Для маленьких — 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )

print(f"Train: {len(X_train):,}")
print(f"Val:   {len(X_val):,}")
print(f"Test:  {len(X_test):,}")


# Модель (без BatchNorm)
class RocketNetESP32(nn.Module):
    def __init__(self, input_dim=11, hidden1=200, hidden2=150, hidden3=80, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.15)

        self.fc3 = nn.Linear(hidden2, hidden3)
        self.ln3 = nn.LayerNorm(hidden3)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden3, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.drop1(self.relu1(self.ln1(self.fc1(x))))
        x = self.drop2(self.relu2(self.ln2(self.fc2(x))))
        x = self.relu3(self.ln3(self.fc3(x)))
        x = self.tanh(self.fc4(x))
        return x


model = RocketNetESP32()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n Модель создана:")
print(f"   Всего параметров: {total_params:,}")
print(f"   Обучаемых: {trainable_params:,}")
print(f"   Размер в памяти: ~{total_params * 4 / 1024:.0f} КБ (float32)")

# Датасеты и загрузчики
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

# Большой батч (чуть ускоряет обучение на CPU, так как видеокарты NVidia у меня отсутсвуют)
BATCH_SIZE = 4096
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n Batch size: {BATCH_SIZE}")
print(f"   Train batches: {len(train_loader)}")

# Обучение
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
)

device = torch.device('cpu')
model.to(device)
print(f"\n Устройство: {device}")

print("\n Начало обучения...")
print("=" * 60)

start_time = time.time()
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
EPOCHS = 100

for epoch in range(EPOCHS):
    #Train
    model.train()
    train_loss = 0
    train_mae = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # Клипинг градиентов
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item() * len(X_batch)
        train_mae += torch.abs(y_pred - y_batch).mean().item() * len(X_batch)

    train_loss /= len(train_dataset)
    train_mae /= len(train_dataset)

    #Val
    model.eval()
    val_loss = 0
    val_mae = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * len(X_batch)
            val_mae += torch.abs(y_pred - y_batch).mean().item() * len(X_batch)

    val_loss /= len(val_dataset)
    val_mae /= len(val_dataset)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_mae'].append(train_mae)
    history['val_mae'].append(val_mae)

    scheduler.step(val_loss)

    # Прогресс каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (EPOCHS - epoch - 1)
        print(f"Epoch {epoch + 1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | MAE: {val_mae:.4f} | "
              f"ETA: {eta / 60:.0f}мин")

    # Early stopping
    if val_loss < best_val_loss - 1e-6:  # Требуем реального улучшения
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'features': features,
            'targets': targets,
        }, 'rocket_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= 25:
            print(f" Early stopping на эпохе {epoch + 1} (нет улучшения 25 эпох)")
            break

elapsed = time.time() - start_time
print(f"\n Обучение завершено за {elapsed / 60:.1f} минут")
print(f"   Лучший Val Loss: {best_val_loss:.6f}")

#Оценка на тесте
print("\n Оценка на тестовом наборе:")
print("=" * 60)

checkpoint = torch.load('rocket_best.pth', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_loss = 0
test_mae = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        test_loss += criterion(y_pred, y_batch.to(device)).item() * len(X_batch)
        test_mae += torch.abs(y_pred - y_batch.to(device)).mean().item() * len(X_batch)
        all_preds.append(y_pred.cpu().numpy())
        all_targets.append(y_batch.numpy())

test_loss /= len(test_dataset)
test_mae /= len(test_dataset)

# Денормализуем для понятной метрики
y_pred_denorm = scaler_y.inverse_transform(np.vstack(all_preds))
y_test_denorm = scaler_y.inverse_transform(np.vstack(all_targets))
errors = np.abs(y_pred_denorm - y_test_denorm)
mean_error_rad = np.mean(errors, axis=0)
mean_error_deg = np.rad2deg(mean_error_rad)

print(f"Test Loss (норм.): {test_loss:.6f}")
print(f"Test MAE  (норм.): {test_mae:.4f}")
print(f"\n Точность в градусах (средняя по |ошибка|):")
print(f"   Pitch: {mean_error_deg[0]:.3f}°")
print(f"   Yaw:   {mean_error_deg[1]:.3f}°")
print(f"   Roll:  {mean_error_deg[2]:.3f}°")

# Export to C++ file for ESP32 (.h - headers) (!!!ТОЛЬКО ПРИ ЗАВЕРШЕНИИ ОБУЧЕНИЯ!!! Если обучение приостановили раньше - запустите export.py)
print("\n Экспорт:")
print("=" * 60)


def export_linear_layer_c(f, layer_name, weight, bias):
    """Экспорт одного линейного слоя в C массив"""
    out_features, in_features = weight.shape

    f.write(f"// {layer_name}: Linear({in_features}, {out_features})\n")
    f.write(f"const float {layer_name}_weight[{out_features}][{in_features}] = {{\n")
    for i in range(out_features):
        f.write("    {")
        for j in range(in_features):
            f.write(f"{weight[i, j]:.8f}")
            if j < in_features - 1:
                f.write(", ")
        f.write("}")
        if i < out_features - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n\n")

    f.write(f"const float {layer_name}_bias[{out_features}] = {{")
    for i in range(out_features):
        f.write(f"{bias[i]:.8f}")
        if i < out_features - 1:
            f.write(", ")
    f.write("};\n\n")


def export_layernorm_c(f, layer_name, weight, bias, eps=1e-5):
    """Экспорт LayerNorm в C массив"""
    size = len(weight)
    f.write(f"// {layer_name}: LayerNorm({size})\n")
    f.write(f"const float {layer_name}_gamma[{size}] = {{")
    for i in range(size):
        f.write(f"{weight[i]:.8f}")
        if i < size - 1:
            f.write(", ")
    f.write("};\n")

    f.write(f"const float {layer_name}_beta[{size}] = {{")
    for i in range(size):
        f.write(f"{bias[i]:.8f}")
        if i < size - 1:
            f.write(", ")
    f.write("};\n")
    f.write(f"#define {layer_name}_EPS {eps}f\n\n")


# Сохраняем в .h файл
state_dict = model.state_dict()

with open('rocket_model.h', 'w') as f:
    f.write("/*\n")
    f.write(" * Модель управления ракетой для ESP32\n")
    f.write(f" * Параметров: {total_params:,}\n")
    f.write(f" * Входы: {len(features)} ({', '.join(features)})\n")
    f.write(f" * Выходы: {len(targets)} ({', '.join(targets)})\n")
    f.write(" */\n\n")
    f.write("#ifndef ROCKET_MODEL_H\n")
    f.write("#define ROCKET_MODEL_H\n\n")
    f.write("#include <Arduino.h>\n\n")

    # Слои
    export_linear_layer_c(f, "fc1",
                          state_dict['fc1.weight'].numpy(),
                          state_dict['fc1.bias'].numpy())
    export_layernorm_c(f, "ln1",
                       state_dict['ln1.weight'].numpy(),
                       state_dict['ln1.bias'].numpy())

    export_linear_layer_c(f, "fc2",
                          state_dict['fc2.weight'].numpy(),
                          state_dict['fc2.bias'].numpy())
    export_layernorm_c(f, "ln2",
                       state_dict['ln2.weight'].numpy(),
                       state_dict['ln2.bias'].numpy())

    export_linear_layer_c(f, "fc3",
                          state_dict['fc3.weight'].numpy(),
                          state_dict['fc3.bias'].numpy())
    export_layernorm_c(f, "ln3",
                       state_dict['ln3.weight'].numpy(),
                       state_dict['ln3.bias'].numpy())

    export_linear_layer_c(f, "fc4",
                          state_dict['fc4.weight'].numpy(),
                          state_dict['fc4.bias'].numpy())

    # Нормализация
    f.write("// Параметры нормализации входов\n")
    f.write(f"const float scaler_X_mean[{len(features)}] = {{")
    for i, val in enumerate(scaler_X.mean_):
        f.write(f"{val:.8f}")
        if i < len(features) - 1:
            f.write(", ")
    f.write("};\n")

    f.write(f"const float scaler_X_scale[{len(features)}] = {{")
    for i, val in enumerate(scaler_X.scale_):
        f.write(f"{val:.8f}")
        if i < len(features) - 1:
            f.write(", ")
    f.write("};\n\n")

    f.write(f"const float scaler_y_mean[{len(targets)}] = {{")
    for i, val in enumerate(scaler_y.mean_):
        f.write(f"{val:.8f}")
        if i < len(targets) - 1:
            f.write(", ")
    f.write("};\n")

    f.write(f"const float scaler_y_scale[{len(targets)}] = {{")
    for i, val in enumerate(scaler_y.scale_):
        f.write(f"{val:.8f}")
        if i < len(targets) - 1:
            f.write(", ")
    f.write("};\n\n")

    f.write("#endif\n")

print(f" rocket_model.h создан ({os.path.getsize('rocket_model.h') / 1024:.1f} КБ)")

# Сохраняем метаданные
meta = {
    'features': features,
    'targets': targets,
    'num_params': total_params,
    'test_mae_norm': float(test_mae),
    'pitch_error_deg': float(mean_error_deg[0]),
    'yaw_error_deg': float(mean_error_deg[1]),
    'roll_error_deg': float(mean_error_deg[2]),
}

with open('rocket_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f" rocket_meta.json создан")

# График (если есть matplotlib)
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0].plot(history['val_loss'], label='Validation', alpha=0.7)
    axes[0].set_title('Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_mae'], label='Train', alpha=0.7)
    axes[1].plot(history['val_mae'], label='Validation', alpha=0.7)
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print(" training_history.png сохранён")
except ImportError:
    print(" matplotlib не установлен, график пропущен")

# Итоги обучения
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Обучение завершено!")
    print("=" * 60)
    print(f" Время обучения: {elapsed / 60:.1f} мин")
    print(f" Параметров: {total_params:,}")
    print(f" Размер модели: ~{total_params * 4 / 1024:.0f} КБ")
    print(f" Ошибка pitch: {mean_error_deg[0]:.3f}°")
    print(f" Ошибка yaw:   {mean_error_deg[1]:.3f}°")
    print(f" Ошибка roll:  {mean_error_deg[2]:.3f}°")
    print(f"\n Файлы для ESP32:")
    print(f"   1. rocket_model.h  — веса и нормализация")
    print(f"   2. rocket_best.pth — чекпоинт PyTorch")
    print(f"   3. rocket_meta.json — метаданные")
    print("=" * 60)