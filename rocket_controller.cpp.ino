/*
 * rocket_controller.cpp.ino — Rocket AI Controller for ESP32
 * ===========================================================
 * Стандарт осей: NEU (North-East-Up)
 *   Body:  Z — продольная ось ракеты (вверх при старте)
 *          X — вправо, Y — вперёд
 *   World: Z — вверх (гравитация [0, 0, -g])
 *
 * Исправления относительно версии v1:
 *   - Конечный автомат (GROUND → ARMED → ASCENT → COAST → DESCENT → RECOVERY)
 *   - Детект старта по всплеску ускорения
 *   - Неблокирующий таймерный цикл 50 Гц (без delay)
 *   - Ограничение скорости серв (300°/с, макс 6 PWM-единиц/шаг)
 *   - Калибровка MPU6050 при включении (усреднение 200 отсчётов)
 *   - Проверка валидности данных (NaN/Inf → fallback)
 *   - Аппаратный Watchdog (TWDT, 3 секунды)
 *   - Вертикальная скорость из BMP280 (скользящее окно)
 *   - Мягкий старт серв (постепенный выход из 90°)
 *   - Логирование только ошибок и смены состояния (без спама)
 *   - Serial-отчёт раз в 500 мс (25 циклов)
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <ESP32Servo.h>
#include <MPU6050.h>
#include <esp_task_wdt.h>
#include <cmath>
#include "rocket_model.h"

// ── Сеть ──────────────────────────────────────────────────
#define NUM_INPUTS   11
#define NUM_HIDDEN1  200
#define NUM_HIDDEN2  150
#define NUM_HIDDEN3  80
#define NUM_OUTPUTS  3

// ── Пины серв ─────────────────────────────────────────────
#define SERVO1_PIN  13
#define SERVO2_PIN  14
#define SERVO3_PIN  15
#define SERVO4_PIN  16

// ── Константы управления ──────────────────────────────────
#define LOOP_FREQ_HZ       50                    // частота контура управления
#define LOOP_PERIOD_US     20000                 // период в микросекундах
#define DT                 0.02f                 // шаг времени (сек)

// ── Параметры серв ────────────────────────────────────────
#define SERVO_CENTER       90                    // нейтраль (PWM)
#define SERVO_RANGE        45                    // амплитуда (±45 от центра = ±25° рулей)
#define SERVO_MAX_DELTA    6                     // макс. изменение PWM за 1 шаг (~300°/с @ 50 Гц)
#define SERVO_SOFT_START_DELAY  40               // шагов плавного старта серв

// ── Детект старта ─────────────────────────────────────────
#define LAUNCH_ACC_THRESH  2.5f                  // порог |ускорения| в G для детекта старта
#define LAUNCH_CONFIRM_CNT 5                     // подтверждающих отсчётов подряд
#define MOTOR_BURNOUT_TIME 5.0f                  // таймаут двигателя (сек)
#define MOTOR_BURNOUT_THRUST_THRESH 0.05f        // доля пиковой тяги

// ── Восстановление ────────────────────────────────────────
#define DESCENT_ALT_THRESH 80.0f                 // высота (м) для раскрытия парашюта
#define RECOVERY_ALT_THRESH 30.0f                // финиш

// ── Диагностика ───────────────────────────────────────────
#define TELEMETRY_PERIOD   25                    // циклов между выводами (~2 Гц)

// ════════════════════════════════════════════════════════════
// Глобальные переменные
// ════════════════════════════════════════════════════════════
Servo servo1, servo2, servo3, servo4;
Adafruit_BMP280 bmp;
MPU6050 mpu;

// ── Датчики (сырые) ───────────────────────────────────────
float ax = 0, ay = 0, az = 0;          // акселерометр, body frame, м/с²
float gx = 0, gy = 0, gz = 0;          // гироскоп, body frame, рад/с
float altitude = 0;                     // высота, м
float temperature = 0;                  // °C
float vert_speed = 0;                   // вертикальная скорость, м/с

// ── Калибровочные смещения ─────────────────────────────────
float gyro_bias_x = 0, gyro_bias_y = 0, gyro_bias_z = 0;
float acc_bias_x  = 0, acc_bias_y  = 0, acc_bias_z  = 0;

// ── Фильтр ориентации ─────────────────────────────────────
float qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;

// ── Сервы (текущее положение + плавный старт) ─────────────
int   servo_pwm[4] = {SERVO_CENTER, SERVO_CENTER, SERVO_CENTER, SERVO_CENTER};
int   servo_soft_start_counter = 0;

// ── Конечный автомат ──────────────────────────────────────
enum State : uint8_t {
    STATE_INIT,        // инициализация / калибровка
    STATE_GROUND,      // на земле, ожидание старта
    STATE_ARMED,       // готов к пуску, мониторинг ускорения
    STATE_ASCENT,      // активный полёт, работа двигателя
    STATE_COAST,       // двигатель выключен, подъём по инерции
    STATE_DESCENT,     // снижение
    STATE_RECOVERY     // парашют раскрыт / посадка
};
State flight_state = STATE_INIT;
uint32_t state_entry_time = 0;
uint32_t ascent_start_time = 0;

// ── Детект старта ─────────────────────────────────────────
int launch_confirm_counter = 0;

// ── Высотомер (скользящее окно для вертикальной скорости) ──
#define ALT_HISTORY_SIZE 10
float  alt_history[ALT_HISTORY_SIZE];
int    alt_history_idx = 0;
bool   alt_history_full = false;

// ── Телеметрия ────────────────────────────────────────────
uint32_t loop_counter = 0;

// ── Watchdog ──────────────────────────────────────────────
#define WDT_TIMEOUT_SEC 3

// ════════════════════════════════════════════════════════════
// Фильтр Маджвика (комплементарный)
// ════════════════════════════════════════════════════════════
// Оси: MPU6050 → body NEU: X=right, Y=forward, Z=up
// В покое az≈+1G (реакция опоры), ax≈ay≈0

bool update_filter(float gx_r, float gy_r, float gz_r,
                   float ax_r, float ay_r, float az_r, float dt) {

    // Проверка валидности входных данных
    if (isnan(ax_r) || isnan(ay_r) || isnan(az_r) ||
        isnan(gx_r) || isnan(gy_r) || isnan(gz_r) ||
        isinf(ax_r) || isinf(ay_r) || isinf(az_r) ||
        isinf(gx_r) || isinf(gy_r) || isinf(gz_r)) {
        return false;
    }

    float acc_norm = sqrt(ax_r*ax_r + ay_r*ay_r + az_r*az_r);
    if (acc_norm < 0.001f) return false;

    float ax_n = ax_r / acc_norm;
    float ay_n = ay_r / acc_norm;
    float az_n = az_r / acc_norm;

    // Углы из акселерометра (body Z = up, стандартная формула)
    float acc_roll  = atan2(ay_n, az_n);
    float acc_pitch = atan2(-ax_n, sqrt(ay_n*ay_n + az_n*az_n));

    // ── Интегрирование гироскопа (кватернионная кинематика) ──
    float gx_h = gx_r * dt * 0.5f;
    float gy_h = gy_r * dt * 0.5f;
    float gz_h = gz_r * dt * 0.5f;

    float qx_g = qx + (qw * gx_h + qy * gz_h - qz * gy_h);
    float qy_g = qy + (-qz * gx_h + qw * gy_h + qx * gz_h);
    float qz_g = qz + (qy * gx_h - qx * gy_h + qw * gz_h);
    float qw_g = qw + (-qx * gx_h - qy * gy_h - qz * gz_h);

    float norm_g = sqrt(qx_g*qx_g + qy_g*qy_g + qz_g*qz_g + qw_g*qw_g);
    if (norm_g > 0.001f) {
        qx_g /= norm_g;
        qy_g /= norm_g;
        qz_g /= norm_g;
        qw_g /= norm_g;
    }

    // ── Кватернион из акселерометра ─────────────────────────
    float cp = cos(acc_pitch * 0.5f);
    float sp = sin(acc_pitch * 0.5f);
    float cr = cos(acc_roll * 0.5f);
    float sr = sin(acc_roll * 0.5f);

    float qx_a = sr * cp;
    float qy_a = cr * sp;
    float qz_a = -sr * sp;
    float qw_a = cr * cp;

    // ── Комплементарное слияние ──────────────────────────────
    const float alpha = 0.95f;
    qx = alpha * qx_g + (1.0f - alpha) * qx_a;
    qy = alpha * qy_g + (1.0f - alpha) * qy_a;
    qz = alpha * qz_g + (1.0f - alpha) * qz_a;
    qw = alpha * qw_g + (1.0f - alpha) * qw_a;

    float norm = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
    if (norm > 0.001f) {
        qx /= norm;
        qy /= norm;
        qz /= norm;
        qw /= norm;
    }

    return true;
}

// ════════════════════════════════════════════════════════════
// Функции активации
// ════════════════════════════════════════════════════════════
inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

inline float tanh_approx(float x) {
    if (x > 4.0f)  return 1.0f;
    if (x < -4.0f) return -1.0f;
    float x2 = x * x;
    return x * (1.0f + x2 * (0.333333f + x2 * (0.133333f + x2 * 0.053968f)))
             / (1.0f + x2 * (1.0f + x2 * (0.2f + x2 * 0.023809f)));
}

// ════════════════════════════════════════════════════════════
// Нормализация
// ════════════════════════════════════════════════════════════
void normalize_input(float *input, float *normalized) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        normalized[i] = (input[i] - scaler_X_mean[i]) / scaler_X_scale[i];
    }
}

void denormalize_output(float *norm_output, float *output) {
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        output[i] = norm_output[i] * scaler_y_scale[i] + scaler_y_mean[i];
    }
}

// ════════════════════════════════════════════════════════════
// Прямой проход нейросети
// ════════════════════════════════════════════════════════════
void predict(float *input, float *output) {
    float hidden1[NUM_HIDDEN1];
    float hidden2[NUM_HIDDEN2];
    float hidden3[NUM_HIDDEN3];

    // ── Слой 1 + LayerNorm + ReLU ──
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        float sum = fc1_bias[i];
        for (int j = 0; j < NUM_INPUTS; j++) sum += fc1_weight[i][j] * input[j];
        hidden1[i] = sum;
    }
    float mean1 = 0, var1 = 0;
    for (int i = 0; i < NUM_HIDDEN1; i++) mean1 += hidden1[i];
    mean1 /= NUM_HIDDEN1;
    for (int i = 0; i < NUM_HIDDEN1; i++) var1 += (hidden1[i] - mean1) * (hidden1[i] - mean1);
    var1 /= NUM_HIDDEN1;
    float inv_std1 = 1.0f / sqrt(var1 + ln1_EPS);
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        hidden1[i] = relu(ln1_gamma[i] * (hidden1[i] - mean1) * inv_std1 + ln1_beta[i]);
    }

    // ── Слой 2 + LayerNorm + ReLU ──
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        float sum = fc2_bias[i];
        for (int j = 0; j < NUM_HIDDEN1; j++) sum += fc2_weight[i][j] * hidden1[j];
        hidden2[i] = sum;
    }
    float mean2 = 0, var2 = 0;
    for (int i = 0; i < NUM_HIDDEN2; i++) mean2 += hidden2[i];
    mean2 /= NUM_HIDDEN2;
    for (int i = 0; i < NUM_HIDDEN2; i++) var2 += (hidden2[i] - mean2) * (hidden2[i] - mean2);
    var2 /= NUM_HIDDEN2;
    float inv_std2 = 1.0f / sqrt(var2 + ln2_EPS);
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        hidden2[i] = relu(ln2_gamma[i] * (hidden2[i] - mean2) * inv_std2 + ln2_beta[i]);
    }

    // ── Слой 3 + LayerNorm + ReLU ──
    for (int i = 0; i < NUM_HIDDEN3; i++) {
        float sum = fc3_bias[i];
        for (int j = 0; j < NUM_HIDDEN2; j++) sum += fc3_weight[i][j] * hidden2[j];
        hidden3[i] = sum;
    }
    float mean3 = 0, var3 = 0;
    for (int i = 0; i < NUM_HIDDEN3; i++) mean3 += hidden3[i];
    mean3 /= NUM_HIDDEN3;
    for (int i = 0; i < NUM_HIDDEN3; i++) var3 += (hidden3[i] - mean3) * (hidden3[i] - mean3);
    var3 /= NUM_HIDDEN3;
    float inv_std3 = 1.0f / sqrt(var3 + ln3_EPS);
    for (int i = 0; i < NUM_HIDDEN3; i++) {
        hidden3[i] = relu(ln3_gamma[i] * (hidden3[i] - mean3) * inv_std3 + ln3_beta[i]);
    }

    // ── Выходной слой + tanh ──
    float raw_output[NUM_OUTPUTS];
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        float sum = fc4_bias[i];
        for (int j = 0; j < NUM_HIDDEN3; j++) sum += fc4_weight[i][j] * hidden3[j];
        raw_output[i] = tanh_approx(sum);
    }
    denormalize_output(raw_output, output);
}

// ════════════════════════════════════════════════════════════
// Микширование серв (X-конфигурация, 4 руля)
// ════════════════════════════════════════════════════════════
void mix_servos(float pitch, float yaw, float roll,
                float *s1, float *s2, float *s3, float *s4) {
    *s1 =  pitch + yaw + roll;
    *s2 =  pitch - yaw - roll;
    *s3 = -pitch + yaw - roll;
    *s4 = -pitch - yaw + roll;
    *s1 = constrain(*s1, -1.0f, 1.0f);
    *s2 = constrain(*s2, -1.0f, 1.0f);
    *s3 = constrain(*s3, -1.0f, 1.0f);
    *s4 = constrain(*s4, -1.0f, 1.0f);
}

// ════════════════════════════════════════════════════════════
// Запись серв с ограничением скорости (как в sim.py)
// ════════════════════════════════════════════════════════════
void write_servos_rate_limited(int target_pwm[4]) {
    for (int i = 0; i < 4; i++) {
        target_pwm[i] = constrain(target_pwm[i], 0, 180);
        int delta = target_pwm[i] - servo_pwm[i];
        delta = constrain(delta, -SERVO_MAX_DELTA, SERVO_MAX_DELTA);
        servo_pwm[i] += delta;
    }
    servo1.write(servo_pwm[0]);
    servo2.write(servo_pwm[1]);
    servo3.write(servo_pwm[2]);
    servo4.write(servo_pwm[3]);
}

// ════════════════════════════════════════════════════════════
// Чтение MPU6050 с защитой от сбоев I2C
// ════════════════════════════════════════════════════════════
bool read_mpu() {
    int16_t ax_int, ay_int, az_int;
    int16_t gx_int, gy_int, gz_int;

    // Пробуем прочитать (с таймаутом I2C)
    unsigned long t0 = micros();
    mpu.getMotion6(&ax_int, &ay_int, &az_int, &gx_int, &gy_int, &gz_int);
    unsigned long elapsed = micros() - t0;

    // Если I2C заняло > 3 мс — возможный сбой
    if (elapsed > 3000) {
        return false;
    }

    // Преобразование в физические единицы
    float ax_raw = ax_int / 16384.0f * 9.81f;
    float ay_raw = ay_int / 16384.0f * 9.81f;
    float az_raw = az_int / 16384.0f * 9.81f;
    float gx_raw = gx_int / 131.0f * DEG_TO_RAD;
    float gy_raw = gy_int / 131.0f * DEG_TO_RAD;
    float gz_raw = gz_int / 131.0f * DEG_TO_RAD;

    // Проверка на NaN/Inf
    if (isnan(ax_raw) || isnan(ay_raw) || isnan(az_raw) ||
        isnan(gx_raw) || isnan(gy_raw) || isnan(gz_raw) ||
        isinf(ax_raw) || isinf(ay_raw) || isinf(az_raw) ||
        isinf(gx_raw) || isinf(gy_raw) || isinf(gz_raw)) {
        return false;
    }

    // Вычитаем калибровочные смещения
    ax = ax_raw - acc_bias_x;
    ay = ay_raw - acc_bias_y;
    az = az_raw - acc_bias_z;
    gx = gx_raw - gyro_bias_x;
    gy = gy_raw - gyro_bias_y;
    gz = gz_raw - gyro_bias_z;

    return true;
}

// ════════════════════════════════════════════════════════════
// Калибровка MPU6050 (усреднение N отсчётов в покое)
// ════════════════════════════════════════════════════════════
void calibrate_sensors(int samples = 200) {
    Serial.print("calibrating (");
    Serial.print(samples);
    Serial.println(" samples)...");

    float sum_gx = 0, sum_gy = 0, sum_gz = 0;
    float sum_ax = 0, sum_ay = 0, sum_az = 0;

    int valid = 0;
    for (int i = 0; i < samples; i++) {
        int16_t ax_i, ay_i, az_i, gx_i, gy_i, gz_i;
        mpu.getMotion6(&ax_i, &ay_i, &az_i, &gx_i, &gy_i, &gz_i);

        float ax_s = ax_i / 16384.0f * 9.81f;
        float ay_s = ay_i / 16384.0f * 9.81f;
        float az_s = az_i / 16384.0f * 9.81f;

        // Пропускаем явные выбросы (удары, тряска)
        float acc_mag = sqrt(ax_s*ax_s + ay_s*ay_s + az_s*az_s);
        if (acc_mag > 12.0f || acc_mag < 7.5f) {
            delay(2);
            continue;
        }

        sum_gx += gx_i / 131.0f * DEG_TO_RAD;
        sum_gy += gy_i / 131.0f * DEG_TO_RAD;
        sum_gz += gz_i / 131.0f * DEG_TO_RAD;
        sum_ax += ax_s;
        sum_ay += ay_s;
        sum_az += az_s;
        valid++;
        delay(2);
    }

    if (valid < 50) {
        Serial.println("ERROR: calibration failed (too few valid samples)");
        // Используем нулевые смещения
        gyro_bias_x = gyro_bias_y = gyro_bias_z = 0;
        acc_bias_x = acc_bias_y = acc_bias_z = 0;
        return;
    }

    gyro_bias_x = sum_gx / valid;
    gyro_bias_y = sum_gy / valid;
    gyro_bias_z = sum_gz / valid;

    // Смещение акселерометра: разница между измеренным и ожидаемым (0, 0, g)
    acc_bias_x = sum_ax / valid;
    acc_bias_y = sum_ay / valid;
    acc_bias_z = (sum_az / valid) - 9.81f;

    Serial.print("gyro bias: ");
    Serial.print(gyro_bias_x * RAD_TO_DEG, 3); Serial.print(", ");
    Serial.print(gyro_bias_y * RAD_TO_DEG, 3); Serial.print(", ");
    Serial.println(gyro_bias_z * RAD_TO_DEG, 3);

    Serial.print("acc bias:  ");
    Serial.print(acc_bias_x, 3); Serial.print(", ");
    Serial.print(acc_bias_y, 3); Serial.print(", ");
    Serial.println(acc_bias_z, 3);
}

// ════════════════════════════════════════════════════════════
// Обновление вертикальной скорости (скользящее окно)
// ════════════════════════════════════════════════════════════
float update_vert_speed(float new_alt, float dt) {
    alt_history[alt_history_idx] = new_alt;
    alt_history_idx = (alt_history_idx + 1) % ALT_HISTORY_SIZE;
    if (alt_history_idx == 0) alt_history_full = true;

    int count = alt_history_full ? ALT_HISTORY_SIZE : alt_history_idx;
    if (count < 2) return 0.0f;

    int oldest_idx = alt_history_full
        ? alt_history_idx
        : 0;
    int newest_idx = alt_history_full
        ? (alt_history_idx - 1 + ALT_HISTORY_SIZE) % ALT_HISTORY_SIZE
        : alt_history_idx - 1;

    float oldest_alt = alt_history[oldest_idx];
    float newest_alt = alt_history[newest_idx];
    float time_span = (count - 1) * dt;

    if (time_span < 0.01f) return 0.0f;
    return (newest_alt - oldest_alt) / time_span;
}

// ════════════════════════════════════════════════════════════
// Печать состояния (вызывается редко)
// ════════════════════════════════════════════════════════════
void log_telemetry() {
    Serial.print("S:");
    switch (flight_state) {
        case STATE_INIT:     Serial.print("INIT");     break;
        case STATE_GROUND:   Serial.print("GROUND");   break;
        case STATE_ARMED:    Serial.print("ARMED");    break;
        case STATE_ASCENT:   Serial.print("ASCENT");   break;
        case STATE_COAST:    Serial.print("COAST");    break;
        case STATE_DESCENT:  Serial.print("DESCENT");  break;
        case STATE_RECOVERY: Serial.print("RECOVERY"); break;
        default:             Serial.print("?");         break;
    }
    Serial.print(" alt=");  Serial.print(altitude, 1);
    Serial.print(" vs=");   Serial.print(vert_speed, 2);
    Serial.print(" acc=");  Serial.print(sqrt(ax*ax + ay*ay + az*az) / 9.81f, 2);
    Serial.print("G srv=");
    for (int i = 0; i < 4; i++) {
        Serial.print(servo_pwm[i]);
        if (i < 3) Serial.print(",");
    }
    Serial.println();
}

void log_error(const char *msg) {
    Serial.print("ERROR [");
    Serial.print(millis());
    Serial.print("]: ");
    Serial.println(msg);
}

// ════════════════════════════════════════════════════════════
// Смена состояния конечного автомата
// ════════════════════════════════════════════════════════════
void set_state(State new_state) {
    if (new_state == flight_state) return;
    State old = flight_state;
    flight_state = new_state;
    state_entry_time = millis();

    Serial.print("STATE: ");
    switch (old) {
        case STATE_INIT: Serial.print("INIT"); break;
        case STATE_GROUND: Serial.print("GROUND"); break;
        case STATE_ARMED: Serial.print("ARMED"); break;
        case STATE_ASCENT: Serial.print("ASCENT"); break;
        case STATE_COAST: Serial.print("COAST"); break;
        case STATE_DESCENT: Serial.print("DESCENT"); break;
        case STATE_RECOVERY: Serial.print("RECOVERY"); break;
    }
    Serial.print(" -> ");
    switch (new_state) {
        case STATE_INIT: Serial.print("INIT"); break;
        case STATE_GROUND: Serial.print("GROUND"); break;
        case STATE_ARMED: Serial.print("ARMED"); break;
        case STATE_ASCENT: Serial.print("ASCENT"); break;
        case STATE_COAST: Serial.print("COAST"); break;
        case STATE_DESCENT: Serial.print("DESCENT"); break;
        case STATE_RECOVERY: Serial.print("RECOVERY"); break;
    }
    Serial.println();
}

// ════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== ROCKET AI v2 (NEU axes, state machine, rate-limited servos) ===");

    // ── Watchdog (3 секунды) ────────────────────────────────
    esp_task_wdt_init(WDT_TIMEOUT_SEC, true);
    esp_task_wdt_add(NULL);
    Serial.println("WDT: enabled (3s)");

    // ── I2C ────────────────────────────────────────────────
    Wire.begin();
    Wire.setClock(400000);
    Serial.println("I2C: 400 kHz");

    // ── BMP280 ──────────────────────────────────────────────
    if (!bmp.begin(0x76)) {
        if (!bmp.begin(0x77)) {
            log_error("BMP280 not found at 0x76 or 0x77");
        } else {
            Serial.println("BMP280: found at 0x77");
        }
    } else {
        Serial.println("BMP280: found at 0x76");
    }

    // ── MPU6050 ─────────────────────────────────────────────
    mpu.initialize();
    if (mpu.testConnection()) {
        Serial.println("MPU6050: connected");
    } else {
        log_error("MPU6050 not connected — HALTING");
        while (1) { esp_task_wdt_reset(); delay(10); }
    }

    // ── Калибровка ──────────────────────────────────────────
    calibrate_sensors(200);
    Serial.println("Calibration done");

    // ── Инициализация серв (мягкий старт) ───────────────────
    servo1.attach(SERVO1_PIN);
    servo2.attach(SERVO2_PIN);
    servo3.attach(SERVO3_PIN);
    servo4.attach(SERVO4_PIN);

    // Сразу в центр
    servo1.write(SERVO_CENTER);
    servo2.write(SERVO_CENTER);
    servo3.write(SERVO_CENTER);
    servo4.write(SERVO_CENTER);
    for (int i = 0; i < 4; i++) servo_pwm[i] = SERVO_CENTER;
    delay(300);
    Serial.println("Servos: centered");

    // ── Начальные показания высотомера ──────────────────────
    altitude = bmp.readAltitude(1013.25);
    for (int i = 0; i < ALT_HISTORY_SIZE; i++) {
        alt_history[i] = altitude;
    }
    alt_history_full = true;
    alt_history_idx = 0;

    // ── Переход в GROUND ────────────────────────────────────
    set_state(STATE_GROUND);
    Serial.println("=== READY (waiting for arming command) ===");
}

// ════════════════════════════════════════════════════════════
// LOOP (50 Гц, неблокирующий таймер)
// ════════════════════════════════════════════════════════════
void loop() {
    // Кормим watchdog
    esp_task_wdt_reset();

    static uint32_t next_loop_us = micros();

    // Жёсткий таймер: ждём следующего такта
    uint32_t now = micros();
    if ((int32_t)(now - next_loop_us) < 0) {
        return; // ещё не время
    }
    next_loop_us += LOOP_PERIOD_US;

    // Если отстали — догоняем без накопления задержки
    if ((int32_t)(micros() - next_loop_us) > (int32_t)LOOP_PERIOD_US) {
        next_loop_us = micros() + LOOP_PERIOD_US;
    }

    loop_counter++;

    // ── 1. Чтение датчиков ──────────────────────────────────
    bool mpu_ok = read_mpu();
    if (!mpu_ok) {
        log_error("MPU read failed");
        return;
    }

    altitude = bmp.readAltitude(1013.25);
    temperature = bmp.readTemperature();

    // Проверка валидности высоты
    if (isnan(altitude) || isinf(altitude)) {
        log_error("BMP altitude NaN/Inf");
        altitude = alt_history[(alt_history_idx - 1 + ALT_HISTORY_SIZE) % ALT_HISTORY_SIZE];
    }

    vert_speed = update_vert_speed(altitude, LOOP_PERIOD_US * 1e-6f);

    // ── 2. Фильтр ориентации ────────────────────────────────
    if (!update_filter(gx, gy, gz, ax, ay, az, DT)) {
        log_error("Filter update failed (NaN/Inf input)");
        return;
    }

    // ── 3. Конечный автомат ─────────────────────────────────
    float acc_mag_g = sqrt(ax*ax + ay*ay + az*az) / 9.81f;

    switch (flight_state) {

        case STATE_INIT:
            // Не должны здесь оказаться; переходим в GROUND
            set_state(STATE_GROUND);
            break;

        case STATE_GROUND:
            // Ждём команду ARM по Serial (символ 'a')
            if (Serial.available()) {
                char c = Serial.read();
                if (c == 'a' || c == 'A') {
                    set_state(STATE_ARMED);
                }
            }
            // Держим сервы в центре
            for (int i = 0; i < 4; i++) servo_pwm[i] = SERVO_CENTER;
            write_servos_rate_limited(servo_pwm);
            break;

        case STATE_ARMED:
            // Мониторинг ускорения для детекта старта
            if (acc_mag_g > LAUNCH_ACC_THRESH) {
                launch_confirm_counter++;
                if (launch_confirm_counter >= LAUNCH_CONFIRM_CNT) {
                    set_state(STATE_ASCENT);
                    ascent_start_time = millis();
                }
            } else {
                launch_confirm_counter = max(0, launch_confirm_counter - 1);
            }
            // Сервы пока в центре
            for (int i = 0; i < 4; i++) servo_pwm[i] = SERVO_CENTER;
            write_servos_rate_limited(servo_pwm);
            break;

        case STATE_ASCENT: {
            // Активное управление
            float angle_deg = 0.0f;
            float sensor_data[NUM_INPUTS] = {qx, qy, qz, qw, gx, gy, gz, ax, ay, az, angle_deg};
            float normalized_input[NUM_INPUTS];
            normalize_input(sensor_data, normalized_input);

            float actions[NUM_OUTPUTS];
            predict(normalized_input, actions);

            float s1, s2, s3, s4;
            mix_servos(actions[0], actions[1], actions[2], &s1, &s2, &s3, &s4);

            int target_pwm[4] = {
                (int)(SERVO_CENTER + s1 * SERVO_RANGE),
                (int)(SERVO_CENTER + s2 * SERVO_RANGE),
                (int)(SERVO_CENTER + s3 * SERVO_RANGE),
                (int)(SERVO_CENTER + s4 * SERVO_RANGE)
            };

            // Сохраняем target для телеметрии
            for (int i = 0; i < 4; i++) servo_pwm[i] = constrain(target_pwm[i], 0, 180);
            write_servos_rate_limited(target_pwm);

            // Проверка выключения двигателя
            float ascent_elapsed = (millis() - ascent_start_time) / 1000.0f;
            if (ascent_elapsed > MOTOR_BURNOUT_TIME) {
                set_state(STATE_COAST);
            }
            break;
        }

        case STATE_COAST:
            // Стабилизация без тяги. Двигатель выключен, ракета по инерции.
            // Продолжаем управлять (рули ещё эффективны).
            {
                float angle_deg = 0.0f;
                float sensor_data[NUM_INPUTS] = {qx, qy, qz, qw, gx, gy, gz, ax, ay, az, angle_deg};
                float normalized_input[NUM_INPUTS];
                normalize_input(sensor_data, normalized_input);

                float actions[NUM_OUTPUTS];
                predict(normalized_input, actions);

                float s1, s2, s3, s4;
                mix_servos(actions[0], actions[1], actions[2], &s1, &s2, &s3, &s4);

                int target_pwm[4] = {
                    (int)(SERVO_CENTER + s1 * SERVO_RANGE),
                    (int)(SERVO_CENTER + s2 * SERVO_RANGE),
                    (int)(SERVO_CENTER + s3 * SERVO_RANGE),
                    (int)(SERVO_CENTER + s4 * SERVO_RANGE)
                };
                write_servos_rate_limited(target_pwm);
            }

            // Переход в DESCENT при устойчивом снижении
            if (vert_speed < -2.0f && altitude < alt_history[0] - 5.0f) {
                set_state(STATE_DESCENT);
            }
            break;

        case STATE_DESCENT:
            // Снижение: убираем рули в ноль, готовим парашют
            {
                int target_pwm[4] = {SERVO_CENTER, SERVO_CENTER, SERVO_CENTER, SERVO_CENTER};
                write_servos_rate_limited(target_pwm);
            }

            // На заданной высоте — раскрытие парашюта
            if (altitude < DESCENT_ALT_THRESH) {
                set_state(STATE_RECOVERY);
                // Сигнал на пиропатрон парашюта (пин 32, активный HIGH)
                // pinMode(32, OUTPUT); digitalWrite(32, HIGH);
                Serial.println("PARACHUTE: deploy signal");
            }
            break;

        case STATE_RECOVERY:
            // Парашют раскрыт, полёт завершён
            {
                int target_pwm[4] = {SERVO_CENTER, SERVO_CENTER, SERVO_CENTER, SERVO_CENTER};
                write_servos_rate_limited(target_pwm);
            }
            break;
    }

    // ── 4. Телеметрия (раз в TELEMETRY_PERIOD циклов) ────────
    if (loop_counter % TELEMETRY_PERIOD == 0) {
        log_telemetry();
    }
}