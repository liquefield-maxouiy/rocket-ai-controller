/*
 * rocket_controller.cpp.ino — Rocket AI Controller for ESP32
 * ===========================================================
 * Стандарт осей: NEU (North-East-Up)
 *   Body:  Z — продольная ось ракеты (вверх при старте)
 *          X — вправо, Y — вперёд
 *   World: Z — вверх (гравитация [0, 0, -g])
 *
 * Исправления v3 (относительно v2):
 *   [FIX#1] Гироскопическая кинематика: исправлены знаки в qy_g и qz_g
 *           (были перепутаны, что давало неверную эволюцию кватерниона).
 *   [FIX#2] Конечный автомат:
 *           - ASCENT→COAST: добавлен детект по спаду тяги (thrust burnout),
 *             таймер — только как резервный上限.
 *           - COAST→DESCENT: сохранена опорная высота земли (ground_alt),
 *             добавлен детект апогея (apogee detected) и подтверждение
 *             снижения (несколько отсчётов подряд).
 *           - ARMED: добавлен таймаут 120 с (возврат в GROUND при отсутствии
 *             старта).
 *           - Все переходы защищены счётчиками подтверждения.
 *   [FIX#3] Сервы:
 *           - Убран сброс servo_pwm перед вызовом rate-limiter (из-за него
 *             ограничение скорости не работало в ASCENT).
 *           - SERVO_MAX_DELTA снижен с 6 до 4 (~200°/с mech), чтобы
 *             сохранить плавность без потери управляемости.
 *   [FIX#4] Безопасность:
 *           - При отказе BMP280 — аварийный переход в RECOVERY с раскрытием
 *             парашюта (fall-safe).
 *           - Добавлен медианный фильтр барометра (3 отсчёта) против выбросов.
 *           - Окно vertical speed увеличено до 25 отсчётов (0.5 с).
 *   [FIX#5] Парашют:
 *           - Раскомментирован и доработан код управления пиропатроном (пин 32).
 *           - One-shot защита: импульс 1 с, однократно.
 *           - pinMode вынесен в setup().
 *   [FIX#6] Добавлен детект посадки (вертикальная скорость ≈ 0 в RECOVERY).
 *   [FIX#7] Watchdog сброс теперь гарантирован даже при возврате из loop
 *           (вынесен в начало, до таймерного гейта).
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <ESP32Servo.h>
#include <MPU6050.h>
#include <esp_task_wdt.h>
#include <cmath>
#include <SD.h>
#include "rocket_model.h"

// ── Сеть ──────────────────────────────────────────────────
#define NUM_INPUTS   11
#define NUM_HIDDEN1  200
#define NUM_HIDDEN2  150
#define NUM_HIDDEN3  80
#define NUM_OUTPUTS  3

// ── Пины ──────────────────────────────────────────────────
#define SERVO1_PIN      13
#define SERVO2_PIN      14
#define SERVO3_PIN      15
#define SERVO4_PIN      16
#define PARACHUTE_PIN   32    // [FIX#5] пиропатрон основного парашюта
#define PARACHUTE2_PIN  33    // [FIX#5] резервный пиропатрон (опционально)

// ── Константы управления ──────────────────────────────────
#define LOOP_FREQ_HZ       50
#define LOOP_PERIOD_US     20000
#define DT                 0.02f

// ── Параметры серв ────────────────────────────────────────
#define SERVO_CENTER       90
#define SERVO_RANGE        45
// [FIX#3] Уменьшено с 6 до 4: ~200°/с механических при 50 Гц
// (4 PWM-ед * (25°/45 PWM-ед) / 0.02 с ≈ 111 °/с на рулях —
//  достаточно быстро, но без рывков)
#define SERVO_MAX_DELTA    4

// ── Детект старта ─────────────────────────────────────────
#define LAUNCH_ACC_THRESH       2.5f
#define LAUNCH_CONFIRM_CNT      5
#define MOTOR_BURNOUT_TIME      8.0f   // [FIX#2] увеличен как резервный上限
#define MOTOR_BURNOUT_THRUST    0.15f  // [FIX#2] доля пиковой тяги для детекта отсечки
#define BURNOUT_CONFIRM_CNT     10     // [FIX#2] подтверждений спада тяги

// ── Восстановление ────────────────────────────────────────
#define DESCENT_ALT_THRESH      80.0f   // высота раскрытия основного парашюта
#define RECOVERY_ALT_THRESH     30.0f   // [FIX#5] резервный парашют
#define LANDING_VS_THRESH       0.5f    // [FIX#6] порог вертикальной скорости посадки
#define ARMING_TIMEOUT_SEC      120     // [FIX#2] таймаут ARMED → GROUND

// ── SD-карта ──────────────────────────────────────────────
#define SD_CS_PIN              4       // пин CS/SS для SD-модуля (обычно 5 на ESP32)
#define SD_LOG_FLUSH_INTERVAL  50      // сбрасывать буфер SD каждые 50 циклов (~1 с)

// ── Диагностика ───────────────────────────────────────────
#define TELEMETRY_PERIOD        25      // ~2 Гц

// ════════════════════════════════════════════════════════════
// Глобальные переменные
// ════════════════════════════════════════════════════════════
Servo servo1, servo2, servo3, servo4;
Adafruit_BMP280 bmp;
MPU6050 mpu;

// ── Флаги оборудования ────────────────────────────────────
bool bmp_ok = false;    // [FIX#4] флаг работоспособности барометра

// ── Датчики (сырые) ───────────────────────────────────────
float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;
float altitude = 0;
float temperature = 0;
float vert_speed = 0;

// ── Калибровочные смещения ─────────────────────────────────
float gyro_bias_x = 0, gyro_bias_y = 0, gyro_bias_z = 0;
float acc_bias_x  = 0, acc_bias_y  = 0, acc_bias_z  = 0;

// ── Фильтр ориентации ─────────────────────────────────────
float qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;

// ── Сервы ─────────────────────────────────────────────────
int servo_pwm[4] = {SERVO_CENTER, SERVO_CENTER, SERVO_CENTER, SERVO_CENTER};

// ── Конечный автомат ──────────────────────────────────────
enum State : uint8_t {
    STATE_INIT,
    STATE_GROUND,
    STATE_ARMED,
    STATE_ASCENT,
    STATE_COAST,
    STATE_DESCENT,
    STATE_RECOVERY
};
State flight_state = STATE_INIT;
uint32_t state_entry_time = 0;
uint32_t ascent_start_time = 0;

// ── Детект старта / отсечки / апогея ──────────────────────
int launch_confirm_counter = 0;
int burnout_confirm_counter = 0;       // [FIX#2] счётчик подтверждения отсечки
int descent_confirm_counter = 0;       // [FIX#2] счётчик подтверждения снижения
float peak_thrust = 1.0f;              // [FIX#2] пиковая перегрузка в ASCENT
float peak_altitude = 0.0f;            // [FIX#2] апогей (максимальная высота)
float ground_altitude = 0.0f;          // [FIX#2] высота земли (сохраняется при старте)
bool  apogee_detected = false;         // [FIX#2] флаг детекта апогея
bool  chute1_deployed = false;         // [FIX#5] флаг раскрытия основного парашюта
bool  chute2_deployed = false;         // [FIX#5] флаг раскрытия резервного парашюта
uint32_t chute1_deploy_time = 0;       // [FIX#5] время подачи импульса

// ── Высотомер (скользящее окно) ───────────────────────────
// [FIX#4] Увеличено с 10 до 25 (0.5 с при 50 Гц) для сглаживания
#define ALT_HISTORY_SIZE 25
float  alt_history[ALT_HISTORY_SIZE];
int    alt_history_idx = 0;
bool   alt_history_full = false;

// ── Медианный фильтр барометра ────────────────────────────
// [FIX#4] Буфер из 3 последних сырых отсчётов для подавления выбросов
#define BMP_MEDIAN_SIZE 3
float  bmp_raw_buf[BMP_MEDIAN_SIZE];
int    bmp_raw_idx = 0;
bool   bmp_raw_full = false;

// ── Телеметрия ────────────────────────────────────────────
uint32_t loop_counter = 0;

// ── Watchdog ──────────────────────────────────────────────
#define MY_WDT_TIMEOUT_SEC 3

// ── SD-карта ──────────────────────────────────────────────
File sd_file;
bool sd_ok = false;
uint32_t sd_flush_counter = 0;
char  sd_log_name[32];

// ── Вспомогательные функции для дублирования логов в Serial + SD ──
void sd_print(const char *s) {
    Serial.print(s);
    if (sd_ok) sd_file.print(s);
}

void sd_println(const char *s) {
    Serial.println(s);
    if (sd_ok) sd_file.println(s);
}

void sd_printf(const char *prefix, float val, int decimals) {
    Serial.print(prefix);
    Serial.print(val, decimals);
    if (sd_ok) { sd_file.print(prefix); sd_file.print(val, decimals); }
}

void sd_printf(const char *prefix, int val) {
    Serial.print(prefix);
    Serial.print(val);
    if (sd_ok) { sd_file.print(prefix); sd_file.print(val); }
}

void sd_printf(const char *prefix, const char *val) {
    Serial.print(prefix);
    Serial.print(val);
    if (sd_ok) { sd_file.print(prefix); sd_file.print(val); }
}

void sd_println_prefix(const char *prefix, float val, int decimals) {
    Serial.print(prefix);
    Serial.println(val, decimals);
    if (sd_ok) { sd_file.print(prefix); sd_file.println(val, decimals); }
}

void sd_println_prefix(const char *prefix, int val) {
    Serial.print(prefix);
    Serial.println(val);
    if (sd_ok) { sd_file.print(prefix); sd_file.println(val); }
}

void sd_print_raw(const char *s) {
    if (sd_ok) sd_file.print(s);
}

void sd_flush() {
    if (sd_ok) sd_file.flush();
}

// ════════════════════════════════════════════════════════════
// Медианный фильтр (для барометра) — [FIX#4]
// ════════════════════════════════════════════════════════════
float median3(float a, float b, float c) {
    if (a > b) { float t = a; a = b; b = t; }
    if (b > c) { float t = b; b = c; c = t; }
    if (a > b) { float t = a; a = b; b = t; }
    return b;
}

float bmp_median_filter(float raw_alt) {
    bmp_raw_buf[bmp_raw_idx] = raw_alt;
    bmp_raw_idx = (bmp_raw_idx + 1) % BMP_MEDIAN_SIZE;
    if (bmp_raw_idx == 0) bmp_raw_full = true;

    if (!bmp_raw_full) return raw_alt;  // ещё не накопили

    return median3(bmp_raw_buf[0], bmp_raw_buf[1], bmp_raw_buf[2]);
}

// ════════════════════════════════════════════════════════════
// Фильтр Маджвика (комплементарный)
// ════════════════════════════════════════════════════════════
// Оси: MPU6050 → body NEU: X=right, Y=forward, Z=up
// В покое az≈+1G (реакция опоры), ax≈ay≈0
//
// [FIX#1] Исправлены знаки в интегрировании гироскопа:
//   Правильный вывод из q̇ = 0.5·q⊗ω (body-rate):
//     q̇x = 0.5·( qw·ωx + qy·ωz − qz·ωy)
//     q̇y = 0.5·( qw·ωy + qz·ωx − qx·ωz)
//     q̇z = 0.5·( qw·ωz + qx·ωy − qy·ωx)
//     q̇w = 0.5·(−qx·ωx − qy·ωy − qz·ωz)

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

    // Углы из акселерометра (body Z = up, NEU)
    // Формулы верны: при Z||g имеем ax≈0, ay≈0, az≈+g →
    //   roll = atan2(0, g) = 0, pitch = atan2(0, g) = 0 ✓
    float acc_roll  = atan2(ay_n, az_n);
    float acc_pitch = atan2(-ax_n, sqrt(ay_n*ay_n + az_n*az_n));

    // ── Интегрирование гироскопа (кватернионная кинематика) ──
    float hx = gx_r * dt * 0.5f;
    float hy = gy_r * dt * 0.5f;
    float hz = gz_r * dt * 0.5f;

    // [FIX#1] Исправлены знаки в qy_g и qz_g:
    //   qy_g: было (−qz·hx + ... + qx·hz) → стало (+qz·hx + ... − qx·hz)
    //   qz_g: было (+qy·hx − qx·hy + ...) → стало (−qy·hx + qx·hy + ...)
    float qx_g = qx + (qw * hx + qy * hz - qz * hy);
    float qy_g = qy + (qw * hy + qz * hx - qx * hz);  // ← исправлено
    float qz_g = qz + (qw * hz + qx * hy - qy * hx);  // ← исправлено
    float qw_g = qw + (-qx * hx - qy * hy - qz * hz);

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

    // ── Комплементарное слияние (LERP + нормализация) ────────
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
// Знаки соответствуют стандартной X-схеме (рули под 45°).
// При реальных испытаниях может потребоваться инвертировать
// отдельные каналы в зависимости от механики.
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
// Запись серв с ограничением скорости
// ════════════════════════════════════════════════════════════
// [FIX#3] Функция больше не требует, чтобы servo_pwm был
// предварительно установлен в target — она сама ведёт
// внутреннее состояние servo_pwm[] к target с rate limiting.
// Вызывающий код НЕ должен перезаписывать servo_pwm[] перед вызовом.
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

    unsigned long t0 = micros();
    mpu.getMotion6(&ax_int, &ay_int, &az_int, &gx_int, &gy_int, &gz_int);
    unsigned long elapsed = micros() - t0;

    if (elapsed > 3000) {
        return false;
    }

    float ax_raw = ax_int / 16384.0f * 9.81f;
    float ay_raw = ay_int / 16384.0f * 9.81f;
    float az_raw = az_int / 16384.0f * 9.81f;
    float gx_raw = gx_int / 131.0f * DEG_TO_RAD;
    float gy_raw = gy_int / 131.0f * DEG_TO_RAD;
    float gz_raw = gz_int / 131.0f * DEG_TO_RAD;

    if (isnan(ax_raw) || isnan(ay_raw) || isnan(az_raw) ||
        isnan(gx_raw) || isnan(gy_raw) || isnan(gz_raw) ||
        isinf(ax_raw) || isinf(ay_raw) || isinf(az_raw) ||
        isinf(gx_raw) || isinf(gy_raw) || isinf(gz_raw)) {
        return false;
    }

    ax = ax_raw - acc_bias_x;
    ay = ay_raw - acc_bias_y;
    az = az_raw - acc_bias_z;
    gx = gx_raw - gyro_bias_x;
    gy = gy_raw - gyro_bias_y;
    gz = gz_raw - gyro_bias_z;

    return true;
}

// ════════════════════════════════════════════════════════════
// Калибровка MPU6050
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
        gyro_bias_x = gyro_bias_y = gyro_bias_z = 0;
        acc_bias_x = acc_bias_y = acc_bias_z = 0;
        return;
    }

    gyro_bias_x = sum_gx / valid;
    gyro_bias_y = sum_gy / valid;
    gyro_bias_z = sum_gz / valid;

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
// [FIX#4] Окно 25 отсчётов = 0.5 с при 50 Гц
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
// Раскрытие парашюта (one-shot) — [FIX#5]
// ════════════════════════════════════════════════════════════
void deploy_parachute_main() {
    if (chute1_deployed) return;
    chute1_deployed = true;
    chute1_deploy_time = millis();
    digitalWrite(PARACHUTE_PIN, HIGH);
    Serial.println("PARACHUTE: MAIN deployed (pin 32 HIGH)");
    if (sd_ok) {
        sd_file.println("PARACHUTE: MAIN deployed (pin 32 HIGH)");
        sd_file.flush();
    }
}

void deploy_parachute_backup() {
    if (chute2_deployed) return;
    chute2_deployed = true;
    digitalWrite(PARACHUTE2_PIN, HIGH);
    Serial.println("PARACHUTE: BACKUP deployed (pin 33 HIGH)");
    if (sd_ok) {
        sd_file.println("PARACHUTE: BACKUP deployed (pin 33 HIGH)");
        sd_file.flush();
    }
}

// [FIX#5] Снятие импульса основного парашюта через 1 секунду
void update_parachute_pulse() {
    if (chute1_deployed && !chute2_deployed) {
        // [FIX#5] Импульс 1 с, затем снимаем
        if (millis() - chute1_deploy_time > 1000) {
            digitalWrite(PARACHUTE_PIN, LOW);
        }
    }
}

// ════════════════════════════════════════════════════════════
// Печать состояния
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
    Serial.print(" ch1="); Serial.print(chute1_deployed ? "Y" : "N");
    Serial.print(" ch2="); Serial.print(chute2_deployed ? "Y" : "N");
    Serial.println();
}

// ── Логирование каждого цикла в CSV-строку на SD ──
void log_csv() {
    if (!sd_ok) return;
    uint32_t t = millis();
    uint8_t st = (uint8_t)flight_state;
    float accG = sqrt(ax*ax + ay*ay + az*az) / 9.81f;
    sd_file.print(t);         sd_file.print(',');
    sd_file.print(st);        sd_file.print(',');
    sd_file.print(altitude, 1); sd_file.print(',');
    sd_file.print(vert_speed, 2); sd_file.print(',');
    sd_file.print(accG, 2);   sd_file.print(',');
    sd_file.print(servo_pwm[0]); sd_file.print(',');
    sd_file.print(servo_pwm[1]); sd_file.print(',');
    sd_file.print(servo_pwm[2]); sd_file.print(',');
    sd_file.print(servo_pwm[3]); sd_file.print(',');
    sd_file.print(chute1_deployed ? 'Y' : 'N'); sd_file.print(',');
    sd_file.print(chute2_deployed ? 'Y' : 'N'); sd_file.print(',');
    sd_file.print(qx, 6);     sd_file.print(',');
    sd_file.print(qy, 6);     sd_file.print(',');
    sd_file.print(qz, 6);     sd_file.print(',');
    sd_file.print(qw, 6);     sd_file.print(',');
    sd_file.println(temperature, 1);
}

void log_error(const char *msg) {
    Serial.print("ERROR [");
    Serial.print(millis());
    Serial.print("]: ");
    Serial.println(msg);
    if (sd_ok) {
        sd_file.print("ERROR [");
        sd_file.print(millis());
        sd_file.print("]: ");
        sd_file.println(msg);
    }
}

// ════════════════════════════════════════════════════════════
// Смена состояния конечного автомата
// ════════════════════════════════════════════════════════════
void set_state(State new_state) {
    if (new_state == flight_state) return;
    State old = flight_state;
    flight_state = new_state;
    state_entry_time = millis();

    // Пишем в Serial
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

    // Дублируем на SD
    if (sd_ok) {
        sd_file.print("STATE: ");
        switch (old) {
            case STATE_INIT: sd_file.print("INIT"); break;
            case STATE_GROUND: sd_file.print("GROUND"); break;
            case STATE_ARMED: sd_file.print("ARMED"); break;
            case STATE_ASCENT: sd_file.print("ASCENT"); break;
            case STATE_COAST: sd_file.print("COAST"); break;
            case STATE_DESCENT: sd_file.print("DESCENT"); break;
            case STATE_RECOVERY: sd_file.print("RECOVERY"); break;
        }
        sd_file.print(" -> ");
        switch (new_state) {
            case STATE_INIT: sd_file.print("INIT"); break;
            case STATE_GROUND: sd_file.print("GROUND"); break;
            case STATE_ARMED: sd_file.print("ARMED"); break;
            case STATE_ASCENT: sd_file.print("ASCENT"); break;
            case STATE_COAST: sd_file.print("COAST"); break;
            case STATE_DESCENT: sd_file.print("DESCENT"); break;
            case STATE_RECOVERY: sd_file.print("RECOVERY"); break;
        }
        sd_file.println();
        sd_file.flush();
    }
}

// ════════════════════════════════════════════════════════════
// Выполнение одного шага управления (общий для ASCENT/COAST)
// ════════════════════════════════════════════════════════════
void run_neural_control() {
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
    // [FIX#3] Не перезаписываем servo_pwm[] — rate limiter сам ведёт историю
    write_servos_rate_limited(target_pwm);
}

// ════════════════════════════════════════════════════════════
// Сброс серв в нейтраль (с rate limiting)
// ════════════════════════════════════════════════════════════
void center_servos() {
    int target[4] = {SERVO_CENTER, SERVO_CENTER, SERVO_CENTER, SERVO_CENTER};
    write_servos_rate_limited(target);
}

// ════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== ROCKET AI v3 (fixed kinematics, state machine, median BMP) ===");

    // ── Watchdog (3 секунды) ────────────────────────────────
    esp_task_wdt_config_t wdt_config = {
        .timeout_ms = 3000,
        .trigger_panic = true
    };
    esp_task_wdt_init(&wdt_config);
    esp_task_wdt_add(NULL);
    Serial.println("WDT: enabled (3s)");

    // ── I2C ────────────────────────────────────────────────
    Wire.begin();
    Wire.setClock(400000);
    Serial.println("I2C: 400 kHz");

    // ── BMP280 ──────────────────────────────────────────────
    // [FIX#4] Явно сохраняем флаг работоспособности
    if (!bmp.begin(0x76)) {
        if (!bmp.begin(0x77)) {
            log_error("BMP280 not found at 0x76 or 0x77");
            bmp_ok = false;
        } else {
            Serial.println("BMP280: found at 0x77");
            bmp_ok = true;
        }
    } else {
        Serial.println("BMP280: found at 0x76");
        bmp_ok = true;
    }

    // ── MPU6050 ─────────────────────────────────────────────
    Wire.begin();
    Wire.setClock(400000);

    mpu.initialize();

    bool mpu_ok = false;
    for (uint8_t addr : {0x68, 0x69}) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            mpu_ok = true;
            break;
        }
    }

    if (mpu_ok) {
        Serial.println("MPU6050: initialized and responding");
    } else {
        log_error("MPU6050 not responding - HALTING");
        while (1) { esp_task_wdt_reset(); delay(10); }
    }

    // ── Калибровка ──────────────────────────────────────────
    calibrate_sensors(200);
    Serial.println("Calibration done");

    // ── Инициализация серв ──────────────────────────────────
    servo1.attach(SERVO1_PIN);
    servo2.attach(SERVO2_PIN);
    servo3.attach(SERVO3_PIN);
    servo4.attach(SERVO4_PIN);

    servo1.write(SERVO_CENTER);
    servo2.write(SERVO_CENTER);
    servo3.write(SERVO_CENTER);
    servo4.write(SERVO_CENTER);
    for (int i = 0; i < 4; i++) servo_pwm[i] = SERVO_CENTER;
    delay(300);
    Serial.println("Servos: centered");

    // ── [FIX#5] Инициализация пинов парашютов ───────────────
    pinMode(PARACHUTE_PIN, OUTPUT);
    digitalWrite(PARACHUTE_PIN, LOW);
    pinMode(PARACHUTE2_PIN, OUTPUT);
    digitalWrite(PARACHUTE2_PIN, LOW);
    Serial.println("Parachute pins: initialized (LOW)");

    // ── SD-карта ────────────────────────────────────────────
    pinMode(SD_CS_PIN, OUTPUT);
    digitalWrite(SD_CS_PIN, HIGH);
    delay(10);

    if (!SD.begin(SD_CS_PIN)) {
        Serial.println("SD: not found — logging to Serial only");
        sd_ok = false;
    } else {
        // Формируем имя файла: log_YYYYMMDD_HHMMSS.csv
        uint32_t now_ms = millis();
        snprintf(sd_log_name, sizeof(sd_log_name), "log_%lu.csv", now_ms);
        sd_file = SD.open(sd_log_name, FILE_WRITE);
        if (sd_file) {
            sd_ok = true;
            Serial.print("SD: opened ");
            Serial.println(sd_log_name);
            // CSV-заголовок
            sd_file.println("time_ms,state,alt,vs,accG,srv1,srv2,srv3,srv4,ch1,ch2,qx,qy,qz,qw,temp");
            sd_file.flush();
        } else {
            Serial.println("SD: failed to create log file");
            sd_ok = false;
        }
    }

    // ── Начальные показания высотомера ──────────────────────
    if (bmp_ok) {
        altitude = bmp.readAltitude(1013.25);
    } else {
        altitude = 0.0f;
    }
    // [FIX#2] Сохраняем высоту земли отдельно
    ground_altitude = altitude;

    for (int i = 0; i < ALT_HISTORY_SIZE; i++) {
        alt_history[i] = altitude;
    }
    alt_history_full = true;
    alt_history_idx = 0;

    // Инициализация медианного буфера барометра
    for (int i = 0; i < BMP_MEDIAN_SIZE; i++) {
        bmp_raw_buf[i] = altitude;
    }
    bmp_raw_full = true;
    bmp_raw_idx = 0;

    // ── Переход в GROUND ────────────────────────────────────
    set_state(STATE_GROUND);
    Serial.print("Ground altitude: ");
    Serial.println(ground_altitude, 1);
    Serial.println("=== READY (send 'a' to arm) ===");
}

// ════════════════════════════════════════════════════════════
// LOOP (50 Гц, неблокирующий таймер)
// ════════════════════════════════════════════════════════════
void loop() {
    // [FIX#7] Кормим watchdog ПЕРВЫМ делом — до любых return
    esp_task_wdt_reset();

    static uint32_t next_loop_us = micros();

    uint32_t now = micros();
    if ((int32_t)(now - next_loop_us) < 0) {
        return;
    }
    next_loop_us += LOOP_PERIOD_US;

    if ((int32_t)(micros() - next_loop_us) > (int32_t)LOOP_PERIOD_US) {
        next_loop_us = micros() + LOOP_PERIOD_US;
    }

    loop_counter++;

    // ── 1. Чтение датчиков ──────────────────────────────────
    bool mpu_ok = read_mpu();
    if (!mpu_ok) {
        log_error("MPU read failed");
        // [FIX#4] Не выходим из loop — продолжаем с последними валидными данными
        // (ax,ay,az,gx,gy,gz сохраняют предыдущие значения)
    }

    // [FIX#4] Безопасное чтение барометра с медианным фильтром
    if (bmp_ok) {
        float raw_alt = bmp.readAltitude(1013.25);
        if (isnan(raw_alt) || isinf(raw_alt)) {
            log_error("BMP altitude NaN/Inf — using last valid");
            // altitude сохраняет предыдущее значение
        } else {
            // [FIX#4] Медианный фильтр против одиночных выбросов
            altitude = bmp_median_filter(raw_alt);
        }
    } else {
        // [FIX#4] Барометр неисправен с самого начала —
        //         используем акселерометр для грубой оценки
        //         (двойное интегрирование слишком шумное, но
        //          детект снижения возможен по другим признакам)
        //         Переходим в аварийный режим, если были в полёте.
        if (flight_state >= STATE_ASCENT) {
            log_error("BMP dead in flight — emergency recovery");
            set_state(STATE_RECOVERY);
            deploy_parachute_main();
        }
    }

    temperature = bmp.readTemperature();

    vert_speed = update_vert_speed(altitude, LOOP_PERIOD_US * 1e-6f);

    // ── 2. Фильтр ориентации ────────────────────────────────
    if (!update_filter(gx, gy, gz, ax, ay, az, DT)) {
        log_error("Filter update failed — using previous quaternion");
        // [FIX#4] Не выходим — используем предыдущий кватернион
    }

    // ── 3. Конечный автомат ─────────────────────────────────
    float acc_mag_g = sqrt(ax*ax + ay*ay + az*az) / 9.81f;

    // [FIX#5] Обновляем состояние импульса парашюта
    update_parachute_pulse();

    switch (flight_state) {

        case STATE_INIT:
            set_state(STATE_GROUND);
            break;

        case STATE_GROUND:
            // Ждём команду ARM по Serial
            if (Serial.available()) {
                char c = Serial.read();
                if (c == 'a' || c == 'A') {
                    set_state(STATE_ARMED);
                }
            }
            center_servos();
            break;

        case STATE_ARMED: {
            // [FIX#2] Таймаут: если старта нет дольше ARMING_TIMEOUT_SEC,
            //         возвращаемся в GROUND (безопасность)
            uint32_t armed_elapsed = (millis() - state_entry_time) / 1000;
            if (armed_elapsed > ARMING_TIMEOUT_SEC) {
                log_error("Arming timeout — returning to GROUND");
                set_state(STATE_GROUND);
                launch_confirm_counter = 0;
                break;
            }

            // Детект старта по всплеску ускорения
            if (acc_mag_g > LAUNCH_ACC_THRESH) {
                launch_confirm_counter++;
                if (launch_confirm_counter >= LAUNCH_CONFIRM_CNT) {
                    set_state(STATE_ASCENT);
                    ascent_start_time = millis();
                    // [FIX#2] Сбрасываем переменные детекта отсечки и апогея
                    burnout_confirm_counter = 0;
                    peak_thrust = acc_mag_g;
                    peak_altitude = altitude;
                    apogee_detected = false;
                    descent_confirm_counter = 0;
                }
            } else {
                launch_confirm_counter = max(0, launch_confirm_counter - 1);
            }
            center_servos();
            break;
        }

        case STATE_ASCENT: {
            run_neural_control();

            // [FIX#2] Отслеживаем пиковую перегрузку для детекта отсечки
            if (acc_mag_g > peak_thrust) {
                peak_thrust = acc_mag_g;
            }
            // [FIX#2] Отслеживаем апогей
            if (altitude > peak_altitude) {
                peak_altitude = altitude;
            }

            // ── Детект отсечки двигателя ─────────────────
            // [FIX#2] Основной критерий: спад тяги ниже доли от пика
            //         Резервный критерий: таймер MOTOR_BURNOUT_TIME
            float thrust_ratio = acc_mag_g / peak_thrust;
            float ascent_elapsed = (millis() - ascent_start_time) / 1000.0f;

            bool thrust_dropped = (peak_thrust > LAUNCH_ACC_THRESH &&
                                   thrust_ratio < MOTOR_BURNOUT_THRUST);

            if (thrust_dropped && ascent_elapsed > 1.0f) {
                // Не ранее чем через 1 с (защита от ложного детекта на старте)
                burnout_confirm_counter++;
            } else {
                burnout_confirm_counter = max(0, burnout_confirm_counter - 1);
            }

            if (burnout_confirm_counter >= BURNOUT_CONFIRM_CNT ||
                ascent_elapsed > MOTOR_BURNOUT_TIME) {
                if (ascent_elapsed > MOTOR_BURNOUT_TIME) {
                    Serial.println("ASCENT: burnout by timeout");
                } else {
                    Serial.println("ASCENT: burnout by thrust drop");
                }
                set_state(STATE_COAST);
            }
            break;
        }

        case STATE_COAST: {
            run_neural_control();

            // [FIX#2] Обновляем апогей
            if (altitude > peak_altitude) {
                peak_altitude = altitude;
            }

            // [FIX#2] Детект апогея: высота упала на 3 м от пика
            //         и вертикальная скорость отрицательна
            if (!apogee_detected && peak_altitude > ground_altitude + 10.0f) {
                if (altitude < peak_altitude - 3.0f && vert_speed < -1.0f) {
                    apogee_detected = true;
                    Serial.print("APOGEE: ");
                    Serial.println(peak_altitude, 1);
                }
            }

            // [FIX#2] Переход в DESCENT:
            //         апогей пройден + устойчивое снижение (подтверждение)
            if (apogee_detected && vert_speed < -3.0f) {
                descent_confirm_counter++;
            } else {
                descent_confirm_counter = max(0, descent_confirm_counter - 1);
            }

            // Дополнительно: принудительный переход, если ракета
            // явно ниже земли (барометр мог не зафиксировать апогей)
            bool below_ground = (altitude < ground_altitude - 20.0f &&
                                 vert_speed < -5.0f);

            if (descent_confirm_counter >= 10 || below_ground) {
                if (below_ground) {
                    log_error("COAST→DESCENT forced (altitude below ground)");
                }
                set_state(STATE_DESCENT);
            }
            break;
        }

        case STATE_DESCENT:
        // Снижение: убираем рули в ноль
        center_servos();

        // Раскрываем парашют по вертикальной скорости
        // Если скорость снижения превысила -5 м/с (падает быстрее 5 м/с)
        if (vert_speed < -5.0f) {
            deploy_parachute_main();
            set_state(STATE_RECOVERY);
        }
        break;

        case STATE_RECOVERY:
            // Парашют(ы) раскрыты, рули в центр
            center_servos();

            // [FIX#6] Детект посадки: вертикальная скорость близка к нулю
            //         в течение нескольких секунд после раскрытия
            {
                uint32_t recovery_elapsed = (millis() - state_entry_time) / 1000;
                if (recovery_elapsed > 5 &&
                    fabs(vert_speed) < LANDING_VS_THRESH &&
                    altitude < ground_altitude + 5.0f) {
                    // Ракета на земле — можно отключить парашютный импульс
                    // (уже снят таймером), дополнительных действий не требуется.
                    // Флаг посадки не выставляем явно — просто логируем.
                    static bool landing_reported = false;
                    if (!landing_reported) {
                        landing_reported = true;
                        Serial.println("LANDING: detected (vertical speed ~ 0)");
                    }
                }
            }
            break;
    }

    // ── 4. Телеметрия ──────────────────────────────────────
    if (loop_counter % TELEMETRY_PERIOD == 0) {
        log_telemetry();
    }

    // ── 5. CSV-логирование на SD (каждый цикл) ────────────
    log_csv();

    // ── 6. Периодический сброс буфера SD ──────────────────
    sd_flush_counter++;
    if (sd_flush_counter >= SD_LOG_FLUSH_INTERVAL) {
        sd_flush_counter = 0;
        sd_flush();
    }

    // ── Команды из монитора порта ──
    if (Serial.available()) {
        char c = Serial.read();
        switch (c) {
            case 'a':
            case 'A':
                if (flight_state == STATE_GROUND) {
                    set_state(STATE_ARMED);
                }
                break;

            case 't':
            case 'T':
                // Принудительный переход в ASCENT (для тестов)
                if (flight_state == STATE_ARMED || flight_state == STATE_GROUND) {
                    set_state(STATE_ASCENT);
                    ascent_start_time = millis();
                    Serial.println("TEST: forced ASCENT mode");
                }
                break;

            case 'p':
            case 'P':
                // Ручной тест парашюта
                deploy_parachute_main();
                Serial.println("TEST: parachute deployed");
                break;

            default:
                break;

            case 'd':
            case 'D':
                if (flight_state == STATE_COAST) {
                    set_state(STATE_DESCENT);
                    Serial.println("TEST: forced DESCENT mode");
                }
                break;
        }
    }
}