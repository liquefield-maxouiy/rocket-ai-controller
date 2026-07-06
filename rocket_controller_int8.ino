#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <ESP32Servo.h>
#include <MPU6050.h>
#include "rocket_model_int8.h"

#define NUM_INPUTS   11
#define NUM_HIDDEN1  200
#define NUM_HIDDEN2  150
#define NUM_HIDDEN3  80
#define NUM_OUTPUTS  3

#define SERVO1_PIN  13
#define SERVO2_PIN  14
#define SERVO3_PIN  15
#define SERVO4_PIN  16

#define DT 0.02f

Servo servo1, servo2, servo3, servo4;
Adafruit_BMP280 bmp;
MPU6050 mpu;

float ax, ay, az;
float gx, gy, gz;
float altitude, temperature;

float qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;

// ============================================================
// complementary filter (fast, no external deps)
// ============================================================
void update_filter(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
    float acc_norm = sqrt(ax*ax + ay*ay + az*az);
    if (acc_norm < 0.001) return;
    ax /= acc_norm;
    ay /= acc_norm;
    az /= acc_norm;

    float acc_roll = atan2(ay, az);
    float acc_pitch = atan2(-ax, sqrt(ay*ay + az*az));

    float qx_prev = qx, qy_prev = qy, qz_prev = qz, qw_prev = qw;
    float gx_r = gx * dt * 0.5f;
    float gy_r = gy * dt * 0.5f;
    float gz_r = gz * dt * 0.5f;

    float qx_gyro = qx_prev + (qw_prev * gx_r + qy_prev * gz_r - qz_prev * gy_r);
    float qy_gyro = qy_prev + (-qz_prev * gx_r + qw_prev * gy_r + qx_prev * gz_r);
    float qz_gyro = qz_prev + (qy_prev * gx_r - qx_prev * gy_r + qw_prev * gz_r);
    float qw_gyro = qw_prev + (-qx_prev * gx_r - qy_prev * gy_r - qz_prev * gz_r);

    float norm_gyro = sqrt(qx_gyro*qx_gyro + qy_gyro*qy_gyro + qz_gyro*qz_gyro + qw_gyro*qw_gyro);
    if (norm_gyro > 0.001) {
        qx_gyro /= norm_gyro;
        qy_gyro /= norm_gyro;
        qz_gyro /= norm_gyro;
        qw_gyro /= norm_gyro;
    }

    float cp = cos(acc_pitch * 0.5f);
    float sp = sin(acc_pitch * 0.5f);
    float cr = cos(acc_roll * 0.5f);
    float sr = sin(acc_roll * 0.5f);

    float qx_acc = sr * cp;
    float qy_acc = cr * sp;
    float qz_acc = -sr * sp;
    float qw_acc = cr * cp;

    float alpha = 0.95f;
    qx = alpha * qx_gyro + (1.0f - alpha) * qx_acc;
    qy = alpha * qy_gyro + (1.0f - alpha) * qy_acc;
    qz = alpha * qz_gyro + (1.0f - alpha) * qz_acc;
    qw = alpha * qw_gyro + (1.0f - alpha) * qw_acc;

    float norm = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
    if (norm > 0.001) {
        qx /= norm;
        qy /= norm;
        qz /= norm;
        qw /= norm;
    }
}

inline float relu(float x) { return (x > 0) ? x : 0; }

inline float tanh_approx(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    float x2 = x * x;
    return x * (1.0f + x2 * (0.333333f + x2 * (0.133333f + x2 * 0.053968f))) /
               (1.0f + x2 * (1.0f + x2 * (0.2f + x2 * 0.023809f)));
}

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

// ============================================================
// INT8 FORWARD PASS (fast, low memory)
// ============================================================
void predict_int8(float *input, float *output) {
    int32_t hidden1[NUM_HIDDEN1];
    int32_t hidden2[NUM_HIDDEN2];
    int32_t hidden3[NUM_HIDDEN3];

    // quantize input to int8
    int8_t input_q[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; i++) {
        input_q[i] = (int8_t)(input[i] * 127.0f);
    }

    // layer 1
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        int32_t sum = fc1_bias[i];
        for (int j = 0; j < NUM_INPUTS; j++) {
            sum += fc1_weight[i * NUM_INPUTS + j] * input_q[j];
        }
        hidden1[i] = sum;
    }

    float hidden1_f[NUM_HIDDEN1];
    for (int i = 0; i < NUM_HIDDEN1; i++) {
        float val = hidden1[i] * fc1_scale;
        hidden1_f[i] = (val > 0) ? val : 0;
    }

    // layer 2
    int32_t hidden2_int[NUM_HIDDEN2];
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        int32_t sum = fc2_bias[i];
        for (int j = 0; j < NUM_HIDDEN1; j++) {
            int8_t h1_q = (int8_t)(hidden1_f[j] / fc1_scale * 127.0f);
            sum += fc2_weight[i * NUM_HIDDEN1 + j] * h1_q;
        }
        hidden2_int[i] = sum;
    }

    float hidden2_f[NUM_HIDDEN2];
    for (int i = 0; i < NUM_HIDDEN2; i++) {
        float val = hidden2_int[i] * fc2_scale;
        hidden2_f[i] = (val > 0) ? val : 0;
    }

    // layer 3
    int32_t hidden3_int[NUM_HIDDEN3];
    for (int i = 0; i < NUM_HIDDEN3; i++) {
        int32_t sum = fc3_bias[i];
        for (int j = 0; j < NUM_HIDDEN2; j++) {
            int8_t h2_q = (int8_t)(hidden2_f[j] / fc2_scale * 127.0f);
            sum += fc3_weight[i * NUM_HIDDEN2 + j] * h2_q;
        }
        hidden3_int[i] = sum;
    }

    float hidden3_f[NUM_HIDDEN3];
    for (int i = 0; i < NUM_HIDDEN3; i++) {
        float val = hidden3_int[i] * fc3_scale;
        hidden3_f[i] = (val > 0) ? val : 0;
    }

    // output layer
    int32_t output_int[NUM_OUTPUTS];
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        int32_t sum = fc4_bias[i];
        for (int j = 0; j < NUM_HIDDEN3; j++) {
            int8_t h3_q = (int8_t)(hidden3_f[j] / fc3_scale * 127.0f);
            sum += fc4_weight[i * NUM_HIDDEN3 + j] * h3_q;
        }
        output_int[i] = sum;
    }

    float raw_output[NUM_OUTPUTS];
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        raw_output[i] = tanh_approx(output_int[i] * fc4_scale);
    }

    denormalize_output(raw_output, output);
}

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

void read_mpu() {
    int16_t ax_int, ay_int, az_int;
    int16_t gx_int, gy_int, gz_int;
    mpu.getMotion6(&ax_int, &ay_int, &az_int, &gx_int, &gy_int, &gz_int);
    ax = ax_int / 16384.0 * 9.81;
    ay = ay_int / 16384.0 * 9.81;
    az = az_int / 16384.0 * 9.81;
    gx = gx_int / 131.0;
    gy = gy_int / 131.0;
    gz = gz_int / 131.0;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("rocket ai int8 controller started");

    Wire.begin();
    Wire.setClock(400000);

    if (!bmp.begin(0x76)) {
        Serial.println("bmp280 not found at 0x76");
        if (!bmp.begin(0x77)) Serial.println("bmp280 not found at 0x77");
        else Serial.println("bmp280 found at 0x77");
    } else {
        Serial.println("bmp280 found at 0x76");
    }

    mpu.initialize();
    if (mpu.testConnection()) Serial.println("mpu6050 connected");
    else Serial.println("mpu6050 not connected");

    servo1.attach(SERVO1_PIN);
    servo2.attach(SERVO2_PIN);
    servo3.attach(SERVO3_PIN);
    servo4.attach(SERVO4_PIN);
    servo1.write(90);
    servo2.write(90);
    servo3.write(90);
    servo4.write(90);
    delay(500);
    Serial.println("servos ready");
    Serial.println("all systems ready (int8)");
}

void loop() {
    read_mpu();
    altitude = bmp.readAltitude(1013.25);
    temperature = bmp.readTemperature();

    update_filter(gx, gy, gz, ax, ay, az, DT);

    float angle_deg = 0.0;
    float sensor_data[NUM_INPUTS] = {qx, qy, qz, qw, gx, gy, gz, ax, ay, az, angle_deg};

    float normalized_input[NUM_INPUTS];
    normalize_input(sensor_data, normalized_input);

    float actions[NUM_OUTPUTS];
    predict_int8(normalized_input, actions);

    float pitch = actions[0];
    float yaw   = actions[1];
    float roll  = actions[2];

    float s1, s2, s3, s4;
    mix_servos(pitch, yaw, roll, &s1, &s2, &s3, &s4);

    int servo1_pwm = 90 + (int)(s1 * 45);
    int servo2_pwm = 90 + (int)(s2 * 45);
    int servo3_pwm = 90 + (int)(s3 * 45);
    int servo4_pwm = 90 + (int)(s4 * 45);

    servo1_pwm = constrain(servo1_pwm, 0, 180);
    servo2_pwm = constrain(servo2_pwm, 0, 180);
    servo3_pwm = constrain(servo3_pwm, 0, 180);
    servo4_pwm = constrain(servo4_pwm, 0, 180);

    servo1.write(servo1_pwm);
    servo2.write(servo2_pwm);
    servo3.write(servo3_pwm);
    servo4.write(servo4_pwm);

    Serial.print("pitch "); Serial.print(pitch, 3);
    Serial.print(" yaw "); Serial.print(yaw, 3);
    Serial.print(" roll "); Serial.print(roll, 3);
    Serial.print(" alt "); Serial.print(altitude, 1);
    Serial.print(" temp "); Serial.print(temperature, 1);
    Serial.print(" servos ");
    Serial.print(servo1_pwm); Serial.print(" ");
    Serial.print(servo2_pwm); Serial.print(" ");
    Serial.print(servo3_pwm); Serial.print(" ");
    Serial.println(servo4_pwm);

    delay(20);
}
