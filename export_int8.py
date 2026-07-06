import torch
import numpy as np
import json

print("🚀 Exporting INT8 model for weak MCUs...")

checkpoint = torch.load('rocket_best.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

def quantize_matrix(weight, target_type=np.int8):
    """Квантует матрицу в int8 с автоматическим подбором масштаба"""
    max_val = np.max(np.abs(weight))
    if max_val < 1e-8:
        return weight.astype(target_type), 1.0
    scale = max_val / 127.0
    quantized = np.round(weight / scale).astype(target_type)
    return quantized, scale

def write_array(f, name, array, shape=None, dtype='int8'):
    """Записывает массив в C-стиле"""
    if shape is not None:
        array = array.reshape(shape)
    flat = array.flatten()
    
    if dtype == 'int8':
        f.write(f"const int8_t {name}[{len(flat)}] = {{")
    else:
        f.write(f"const float {name}[{len(flat)}] = {{")
    
    for i, val in enumerate(flat):
        if i % 16 == 0:
            f.write("\n    ")
        if dtype == 'int8':
            f.write(f"{int(val)}")
        else:
            f.write(f"{val:.8f}")
        if i < len(flat) - 1:
            f.write(", ")
    f.write("\n};\n\n")

# ============================================================
# КВАНТУЕМ ВСЕ СЛОИ
# ============================================================
fc1_w, fc1_s = quantize_matrix(state_dict['fc1.weight'].numpy())
fc1_b, _ = quantize_matrix(state_dict['fc1.bias'].numpy())

fc2_w, fc2_s = quantize_matrix(state_dict['fc2.weight'].numpy())
fc2_b, _ = quantize_matrix(state_dict['fc2.bias'].numpy())

fc3_w, fc3_s = quantize_matrix(state_dict['fc3.weight'].numpy())
fc3_b, _ = quantize_matrix(state_dict['fc3.bias'].numpy())

fc4_w, fc4_s = quantize_matrix(state_dict['fc4.weight'].numpy())
fc4_b, _ = quantize_matrix(state_dict['fc4.bias'].numpy())

# ============================================================
# ЭКСПОРТ В .h
# ============================================================
with open('rocket_model_int8.h', 'w') as f:
    f.write("// rocket_model_int8.h - INT8 quantized model for weak MCUs\n")
    f.write("// Generated from rocket_best.pth\n\n")
    f.write("#ifndef ROCKET_MODEL_INT8_H\n")
    f.write("#define ROCKET_MODEL_INT8_H\n\n")
    f.write("#include <Arduino.h>\n\n")

    # Размеры слоёв
    f.write("// Layer sizes\n")
    f.write("#define NUM_INPUTS  11\n")
    f.write("#define NUM_HIDDEN1 200\n")
    f.write("#define NUM_HIDDEN2 150\n")
    f.write("#define NUM_HIDDEN3 80\n")
    f.write("#define NUM_OUTPUTS 3\n\n")

    # Веса (int8)
    f.write("// ============================================================\n")
    f.write("// QUANTIZED WEIGHTS (int8)\n")
    f.write("// ============================================================\n\n")
    
    write_array(f, "fc1_weight", fc1_w, (200, 11), 'int8')
    write_array(f, "fc1_bias", fc1_b, (200,), 'int8')
    
    write_array(f, "fc2_weight", fc2_w, (150, 200), 'int8')
    write_array(f, "fc2_bias", fc2_b, (150,), 'int8')
    
    write_array(f, "fc3_weight", fc3_w, (80, 150), 'int8')
    write_array(f, "fc3_bias", fc3_b, (80,), 'int8')
    
    write_array(f, "fc4_weight", fc4_w, (3, 80), 'int8')
    write_array(f, "fc4_bias", fc4_b, (3,), 'int8')

    # Масштабы (float)
    f.write("// ============================================================\n")
    f.write("// QUANTIZATION SCALES\n")
    f.write("// ============================================================\n\n")
    f.write(f"const float fc1_scale = {fc1_s:.8f}f;\n")
    f.write(f"const float fc2_scale = {fc2_s:.8f}f;\n")
    f.write(f"const float fc3_scale = {fc3_s:.8f}f;\n")
    f.write(f"const float fc4_scale = {fc4_s:.8f}f;\n\n")

    # Scaler (float)
    f.write("// ============================================================\n")
    f.write("// NORMALIZATION (float)\n")
    f.write("// ============================================================\n\n")
    
    features = ['quat_x', 'quat_y', 'quat_z', 'quat_w',
                'omega_x', 'omega_y', 'omega_z',
                'acc_x', 'acc_y', 'acc_z',
                'angle_deg']
    
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

    # Денормализация выходов
    targets = ['action_pitch', 'action_yaw', 'action_roll']
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

print("✅ rocket_model_int8.h generated")
print(f"📊 fc1_scale = {fc1_s:.6f}")
print(f"📊 fc2_scale = {fc2_s:.6f}")
print(f"📊 fc3_scale = {fc3_s:.6f}")
print(f"📊 fc4_scale = {fc4_s:.6f}")
