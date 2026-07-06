import torch
import numpy as np
import json

# Загрузка чекпоинта
checkpoint = torch.load('rocket_best.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Экспорт .h
def export_linear_layer_c(f, layer_name, weight, bias):
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

# Открываем файл на запись
with open('rocket_model.h', 'w') as f:
    f.write("/* Модель управления ракетой для ESP32 */\n\n")
    f.write("#ifndef ROCKET_MODEL_H\n")
    f.write("#define ROCKET_MODEL_H\n\n")
    f.write("#include <Arduino.h>\n\n")

    # Экспортируем слои
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

    # Параметры нормализации
    features = ['quat_x', 'quat_y', 'quat_z', 'quat_w',
                'omega_x', 'omega_y', 'omega_z',
                'acc_x', 'acc_y', 'acc_z', 'angle_deg']

    f.write("const float scaler_X_mean[12] = {")
    for i, val in enumerate(scaler_X.mean_):
        f.write(f"{val:.8f}")
        if i < 11:
            f.write(", ")
    f.write("};\n")

    f.write("const float scaler_X_scale[12] = {")
    for i, val in enumerate(scaler_X.scale_):
        f.write(f"{val:.8f}")
        if i < 11:
            f.write(", ")
    f.write("};\n\n")

    f.write("#endif\n")

print("rocket_model.h создан")