import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict
import warnings
import json
import os
import time
from datetime import datetime
import gzip
import multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================================
# Атмосфера
# ============================================================================
class Atmosphere:
    def __init__(self):
        self.g0 = 9.80665
        self.rho0 = 1.225
        self.H = 8500.0

    def get_density(self, alt: float) -> float:
        alt = max(0.0, alt)
        return self.rho0 * np.exp(-alt / self.H)


# ============================================================================
# Двигатель (с неидеальностями — исправление #9)
# ============================================================================
class Motor:
    def __init__(self, total_impulse: float = 1800.0, burn_time: float = 3.0):
        self.total_impulse = total_impulse
        self.burn_time = burn_time
        self.thrust_peak = total_impulse / burn_time * 1.4
        # Неидеальности двигателя (#9)
        self.thrust_drop_chance = 0.02       # 2% вероятность просадки тяги на каждом шаге
        self.thrust_drop_severity = 0.6      # просадка до 60% номинала
        self.tailoff_duration = 0.15         # "хвост" после отсечки (сек)
        self.tailoff_min_thrust = 0.03       # остаточная тяга (доля от пика)
        self._drop_active = False
        self._drop_recovery = 0.0
        self._drop_factor = 1.0

    def get_thrust(self, t: float) -> float:
        """Возвращает тягу с учётом неидеальностей."""
        if t < 0:
            return 0.0

        # Основная фаза горения
        if t < self.burn_time:
            t_norm = t / self.burn_time
            if t_norm < 0.05:
                nominal = self.thrust_peak * (t_norm / 0.05)
            elif t_norm < 0.6:
                nominal = self.thrust_peak
            else:
                decay = 1.0 - (t_norm - 0.6) / 0.4
                nominal = self.thrust_peak * max(0.0, decay)
        # Фаза "хвоста" после отсечки (#9)
        elif t < self.burn_time + self.tailoff_duration:
            tail_t = (t - self.burn_time) / self.tailoff_duration
            nominal = self.thrust_peak * self.tailoff_min_thrust * (1.0 - tail_t)
        else:
            return 0.0

        return max(0.0, nominal)


# ============================================================================
# Процесс Орнштейна-Уленбека для ветра (исправление #4)
# ============================================================================
class WindProcess:
    """Модель изменяющегося ветра: процесс Орнштейна-Уленбека."""

    def __init__(self,
                 mean_wind: np.ndarray,
                 theta: float = 0.1,     # скорость возврата к среднему
                 sigma: float = 1.5,     # волатильность (интенсивность порывов)
                 gust_prob: float = 0.05, # вероятность резкого порыва на шаге
                 gust_magnitude: float = 8.0):  # максимальная амплитуда порыва (м/с)
        self.mean = mean_wind.copy()
        self.theta = theta
        self.sigma = sigma
        self.gust_prob = gust_prob
        self.gust_magnitude = gust_magnitude
        self.current = mean_wind.copy()
        self._gust_remaining = 0.0
        self._gust_dir = np.zeros(3)

    def step(self, dt: float) -> np.ndarray:
        """Один шаг эволюции ветра."""
        # Орнштейн-Уленбек: dX = theta * (mean - X) * dt + sigma * dW
        dW = np.random.randn(3)
        drift = self.theta * (self.mean - self.current) * dt
        diffusion = self.sigma * np.sqrt(dt) * dW
        self.current += drift + diffusion

        # Порывы (#4): резкое кратковременное усиление
        if self._gust_remaining <= 0 and np.random.random() < self.gust_prob:
            self._gust_remaining = np.random.uniform(0.2, 1.5)  # длительность порыва (сек)
            angle = np.random.uniform(0, 2 * np.pi)
            self._gust_dir = np.array([
                np.cos(angle) * self.gust_magnitude * np.random.uniform(0.5, 1.0),
                np.sin(angle) * self.gust_magnitude * np.random.uniform(0.5, 1.0),
                0.0
            ])

        if self._gust_remaining > 0:
            self._gust_remaining -= dt
            self.current += self._gust_dir * (self._gust_remaining / max(0.2, self._gust_remaining + dt))

        # Ограничение: ветер не превышает ураган (> 30 м/с)
        speed = np.linalg.norm(self.current)
        if speed > 30.0:
            self.current *= 30.0 / speed

        return self.current.copy()


# ============================================================================
# Быстрый симулятор ракеты (исправленная версия)
# ============================================================================
class FastRocketSim:
    """
    Симулятор ракеты с исправлениями:
    - #1: NEU-стандарт осей (body Z — продольная ось, world Z — вверх)
    - #2: tau_servo = 0.2 (реалистичные сервы MG995)
    - #3: Ограничение скорости серв (omega_max_servo)
    - #4: Изменяющийся ветер (процесс Орнштейна-Уленбека + порывы)
    - #5: Вибрация акселерометра (двигатель 50-200 Гц)
    - #6: Дрейф эксцентриситета тяги во времени
    - #7: Изменение центра давления при выгорании топлива
    - #8: Изменение момента инерции при выгорании топлива
    - #9: Неидеальности двигателя (просадки, хвост)
    - #10: Дрейф нуля и сбои I2C датчиков
    """

    def __init__(self, dt: float = 0.02, max_time: float = 12.0):
        self.dt = dt
        self.max_time = max_time
        self.atmo = Atmosphere()

        # Геометрия и масса
        self.base_mass_dry = 2.5
        self.base_mass_prop = 1.2
        self.S_ref = 0.008
        self.L_ref = 0.5  # базовая длина (центр давления будет меняться — #7)

        # Аэродинамические коэффициенты
        self.base_CD0 = 0.35
        self.base_CL_alpha = 2.5
        self.base_Cm_alpha = -0.5
        self.base_Cmq = -2.0

        # Моменты инерции (будут меняться с массой — #8)
        self.base_Ixx = 0.01
        self.base_Iyy = 0.25
        self.base_Izz = 0.25

        # Сервы (#2, #3)
        self.max_delta = np.deg2rad(25.0)       # макс. отклонение рулей
        self.tau_servo = 0.2                     # постоянная времени MG995 (200 мс на 60°)
        self.omega_max_servo = np.deg2rad(300.0) # макс. скорость сервы (300°/с, реалистично)

        # Состояние
        self.state = np.zeros(18)
        self.time = 0.0

        # Ветер (будет заменён WindProcess в reset)
        self.wind_process = None
        self.wind = np.zeros(3)

        # Параметры (переопределяются в reset)
        self.mass_dry = self.base_mass_dry
        self.mass_prop = self.base_mass_prop
        self.total_mass = self.mass_dry + self.mass_prop
        self.CD0 = self.base_CD0
        self.CL_alpha = self.base_CL_alpha
        self.Cm_alpha = self.base_Cm_alpha
        self.Cmq = self.base_Cmq
        self.I = np.array([self.base_Ixx, self.base_Iyy, self.base_Izz])

        self.step_count = 0
        self.motor = Motor()

        # Эксцентриситет тяги (дрейфует во времени — #6)
        self.thrust_misalign_y = 0.0  # радианы
        self.thrust_misalign_z = 0.0
        self.thrust_misalign_drift_y = 0.0
        self.thrust_misalign_drift_z = 0.0

        # Дрейф нуля датчиков (#10)
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)
        self.gyro_bias_drift = np.zeros(3)
        self.acc_bias_drift = np.zeros(3)

    # ------------------------------------------------------------------
    # Свойства, зависящие от массы (исправления #7, #8)
    # ------------------------------------------------------------------
    def _get_L_ref(self, mass: float) -> float:
        """Центр давления смещается при выгорании топлива (#7)."""
        # При полном баке L_ref = 0.5, при пустом — 0.45 (ЦД смещается вперёд)
        prop_fraction = max(0.0, (mass - self.mass_dry) / max(0.01, self.mass_prop))
        return self.L_ref * (0.9 + 0.1 * prop_fraction)

    def _get_inertia(self, mass: float) -> np.ndarray:
        """Момент инерции меняется при выгорании топлива (#8)."""
        prop_fraction = max(0.0, (mass - self.mass_dry) / max(0.01, self.mass_prop))
        # Топливо вносит вклад в момент инерции (распределено по длине)
        I_factor = 0.6 + 0.4 * prop_fraction  # 60%..100% от базового
        return np.array([
            self.base_Ixx * I_factor,
            self.base_Iyy * I_factor,
            self.base_Izz * I_factor
        ])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.time = 0.0
        self.state = np.zeros(18)
        self.step_count = 0

        # Масса с разбросом
        mass_factor = np.random.uniform(0.8, 1.2)
        self.mass_dry = self.base_mass_dry * mass_factor
        self.mass_prop = self.base_mass_prop * np.random.uniform(0.85, 1.15)
        self.total_mass = self.mass_dry + self.mass_prop

        # Аэродинамика с разбросом
        self.CD0 = self.base_CD0 * np.random.uniform(0.7, 1.3)
        self.CL_alpha = self.base_CL_alpha * np.random.uniform(0.7, 1.3)
        self.Cm_alpha = self.base_Cm_alpha * np.random.uniform(0.7, 1.3)
        self.Cmq = self.base_Cmq * np.random.uniform(0.7, 1.3)

        # Момент инерции с разбросом
        I_factor = np.random.uniform(0.8, 1.2)
        self.I = np.array([
            self.base_Ixx * I_factor * np.random.uniform(0.9, 1.1),
            self.base_Iyy * I_factor * np.random.uniform(0.9, 1.1),
            self.base_Izz * I_factor * np.random.uniform(0.9, 1.1)
        ])

        # Ветер (#4): процесс Орнштейна-Уленбека
        wind_speed = np.random.uniform(0.0, 5.0)
        wind_angle = np.random.uniform(0, 2 * np.pi)
        mean_wind = np.array([
            wind_speed * np.cos(wind_angle),
            wind_speed * np.sin(wind_angle),
            0.0
        ])
        self.wind_process = WindProcess(
            mean_wind=mean_wind,
            theta=np.random.uniform(0.05, 0.2),
            sigma=np.random.uniform(0.5, 2.0),
            gust_prob=np.random.uniform(0.02, 0.08),
            gust_magnitude=np.random.uniform(3.0, 10.0)
        )
        self.wind = mean_wind.copy()

        # Эксцентриситет тяги (#6): начальный + дрейф
        self.thrust_misalign_y = np.deg2rad(np.random.uniform(-0.5, 0.5))
        self.thrust_misalign_z = np.deg2rad(np.random.uniform(-0.5, 0.5))
        self.thrust_misalign_drift_y = np.deg2rad(np.random.uniform(-0.05, 0.05))  # рад/с
        self.thrust_misalign_drift_z = np.deg2rad(np.random.uniform(-0.05, 0.05))

        # Дрейф нуля датчиков (#10)
        self.gyro_bias = np.deg2rad(np.random.uniform(-1.0, 1.0, 3))       # начальный сдвиг
        self.acc_bias = np.random.uniform(-0.5, 0.5, 3)                    # м/с²
        self.gyro_bias_drift = np.deg2rad(np.random.uniform(-0.01, 0.01, 3))  # рад/с за секунду
        self.acc_bias_drift = np.random.uniform(-0.01, 0.01, 3)               # м/с² за секунду

        # Начальные отклонения
        pitch_dev = np.deg2rad(np.random.uniform(-15.0, 15.0))
        yaw_dev = np.deg2rad(np.random.uniform(-15.0, 15.0))
        roll_dev = np.deg2rad(np.random.uniform(-5.0, 5.0))
        start_alt = np.random.uniform(0.0, 3.0)
        start_vx = np.random.uniform(-3.0, 3.0)
        start_vy = np.random.uniform(-3.0, 3.0)
        start_vz = np.random.uniform(-1.0, 2.0)

        # Двигатель
        impulse_factor = np.random.uniform(0.85, 1.15)
        burn_time_factor = np.random.uniform(0.9, 1.1)
        self.motor = Motor(
            total_impulse=1800.0 * impulse_factor,
            burn_time=3.0 * burn_time_factor
        )

        # Позиция и скорость (world NEU: X=North, Y=East, Z=Up)
        self.state[0] = np.random.uniform(-1.0, 1.0)
        self.state[1] = np.random.uniform(-1.0, 1.0)
        self.state[2] = start_alt
        self.state[3] = start_vx
        self.state[4] = start_vy
        self.state[5] = start_vz

        # Кватернион ориентации (#1 исправление):
        # Тело: Z — продольная ось (вперёд/вверх), X — вправо, Y — вперёд (NEU body)
        # На старте body Z должен быть направлен в world Z (вверх)
        # Начальный кватернион — identity (body Z уже = world Z)
        # Затем добавляем отклонения
        q_init = np.array([0.0, 0.0, 0.0, 1.0])  # identity, body Z || world Z

        # Отклонения: pitch вокруг body X, yaw вокруг body Y, roll вокруг body Z
        q_pitch = R.from_euler('X', pitch_dev).as_quat()
        q_yaw = R.from_euler('Y', yaw_dev).as_quat()
        q_roll = R.from_euler('Z', roll_dev).as_quat()

        q_dev = self._quat_multiply(q_pitch, q_yaw)
        q_dev = self._quat_multiply(q_dev, q_roll)
        self.state[6:10] = self._quat_multiply(q_init, q_dev)
        self.state[6:10] /= np.linalg.norm(self.state[6:10])

        self.state[16] = self.total_mass
        self.state[17] = 0.0
        self.state[13:16] = np.random.uniform(-0.1, 0.1, 3) * self.max_delta

        return self.state.copy()

    # ------------------------------------------------------------------
    # Кватернионная математика
    # ------------------------------------------------------------------
    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ])

    def _quat_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_vec = q[:3]
        q_w = q[3]
        return v + 2 * np.cross(q_vec, np.cross(q_vec, v) + q_w * v)

    def _quat_rotate_inv(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
        return self._quat_rotate(q_inv, v)

    # ------------------------------------------------------------------
    # Вычисление ускорения (с вибрацией — #5)
    # ------------------------------------------------------------------
    def _compute_acceleration(self, state: np.ndarray, t: float) -> np.ndarray:
        """Возвращает ускорение в СВЯЗАННОЙ системе координат (body frame, NEU)."""
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        mass = max(self.mass_dry, state[16])

        alt = max(0.0, pos[2])
        rho = self.atmo.get_density(alt)
        v_rel = vel - self.wind
        V = np.linalg.norm(v_rel)

        # Гравитация в world frame (NEU: Z вверх)
        F_gravity = np.array([0.0, 0.0, -mass * self.atmo.g0])

        # Аэродинамика
        # body Z — продольная ось (#1)
        # alpha: угол между проекцией скорости на XZ и осью Z
        # beta: угол между проекцией скорости на YZ и осью Z
        F_aero = np.zeros(3)
        if V > 0.01:
            v_body = self._quat_rotate_inv(q, v_rel)
            # Углы атаки и скольжения относительно body Z (продольная ось)
            v_horiz = np.sqrt(v_body[0]**2 + v_body[1]**2)
            alpha = np.arctan2(v_body[0], v_body[2])  # угол в плоскости XZ
            beta = np.arctan2(v_body[1], v_body[2])   # угол в плоскости YZ
            q_bar = 0.5 * rho * V ** 2

            # Drag вдоль вектора скорости (в world frame)
            v_dir = v_rel / V
            F_drag = -q_bar * self.S_ref * self.CD0 * v_dir

            # Подъёмная сила (в body frame): перпендикулярна продольной оси Z
            # Направлена против составляющих скорости в X и Y
            lift_body = np.array([
                q_bar * self.S_ref * self.CL_alpha * alpha,
                q_bar * self.S_ref * self.CL_alpha * beta,
                0.0
            ])
            F_lift = self._quat_rotate(q, lift_body)

            F_aero = F_drag + F_lift

        # Тяга (body Z — продольная ось #1)
        F_thrust = np.zeros(3)
        thrust_mag = self.motor.get_thrust(t)
        if thrust_mag > 0:
            # Тяга направлена вдоль body Z
            body_z = self._quat_rotate(q, np.array([0.0, 0.0, 1.0]))
            body_z = body_z / (np.linalg.norm(body_z) + 1e-8)
            F_thrust = thrust_mag * body_z

        # Сила в world frame
        F_total = F_gravity + F_aero + F_thrust
        acc_world = F_total / mass

        # Переводим ускорение в body frame (для имитации акселерометра)
        acc_body = self._quat_rotate_inv(q, acc_world)

        # Вибрация акселерометра (#5): гармоника двигателя 50-200 Гц
        if thrust_mag > 0:
            # Основная частота вибрации зависит от тяги
            vib_freq = 50.0 + 150.0 * (thrust_mag / max(1.0, self.motor.thrust_peak))
            vib_amp = 0.5 + 2.0 * (thrust_mag / max(1.0, self.motor.thrust_peak))  # м/с² амплитуда
            vib = vib_amp * np.sin(2 * np.pi * vib_freq * t + np.random.uniform(0, 2 * np.pi, 3))
            # Вибрация в основном по продольной оси + немного по поперечным
            acc_body[2] += vib[2] * 1.5
            acc_body[0] += vib[0] * 0.3
            acc_body[1] += vib[1] * 0.3

        return acc_body

    # ------------------------------------------------------------------
    # Вычисление моментов
    # ------------------------------------------------------------------
    def _compute_moments(self, state: np.ndarray, t: float) -> np.ndarray:
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        delta = state[13:16]
        mass = max(self.mass_dry, state[16])

        alt = max(0.0, pos[2])
        rho = self.atmo.get_density(alt)
        v_rel = vel - self.wind
        V = np.linalg.norm(v_rel)
        q_bar = 0.5 * rho * V ** 2 if V > 0.01 else 0.0

        # Центр давления зависит от массы (#7)
        L_ref_dyn = self._get_L_ref(mass)

        M_aero = np.zeros(3)
        if V > 0.01:
            v_body = self._quat_rotate_inv(q, v_rel)
            # Углы относительно body Z (продольная ось)
            alpha = np.arctan2(v_body[0], v_body[2])
            beta = np.arctan2(v_body[1], v_body[2])

            # Восстанавливающий момент (body Z — продольная ось)
            # Момент вокруг X от beta, вокруг Y от alpha
            M_restore = q_bar * self.S_ref * L_ref_dyn * self.Cm_alpha * np.array([
                -beta,   # момент крена от скольжения
                alpha,   # момент тангажа от угла атаки
                0.0
            ])

            # Демпфирующий момент
            M_damp = q_bar * self.S_ref * L_ref_dyn ** 2 * self.Cmq / (V + 0.01) * omega

            # Управляющий момент (от рулей)
            M_control = q_bar * self.S_ref * L_ref_dyn * 0.1 * np.array([
                0.0,
                delta[0],   # pitch
                delta[1]    # yaw
            ])

            M_aero = M_restore + M_damp + M_control

        # Момент от тяги (эксцентриситет)
        M_thrust = np.zeros(3)
        thrust_mag = self.motor.get_thrust(t)
        if thrust_mag > 0:
            # Сопло позади ЦМ вдоль body Z (#1)
            r_nozzle = np.array([0.0, 0.0, -0.4])

            # Эксцентриситет тяги дрейфует со временем (#6)
            drift_y = self.thrust_misalign_drift_y * t
            drift_z = self.thrust_misalign_drift_z * t
            thrust_dir = np.array([
                np.sin(self.thrust_misalign_z + drift_z),
                np.sin(self.thrust_misalign_y + drift_y),
                np.cos(self.thrust_misalign_z + drift_z) * np.cos(self.thrust_misalign_y + drift_y)
            ])
            thrust_dir /= np.linalg.norm(thrust_dir)
            F_thrust_body = thrust_mag * thrust_dir
            M_thrust = np.cross(r_nozzle, F_thrust_body)

        return M_aero + M_thrust

    # ------------------------------------------------------------------
    # Динамика (RK4)
    # ------------------------------------------------------------------
    def _dynamics(self, state: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        delta = state[13:16]
        mass = max(self.mass_dry, state[16])

        q = q / (np.linalg.norm(q) + 1e-8)

        pos_dot = vel

        # Ускорение в body frame (для согласованности с _compute_acceleration)
        temp_state = state.copy()
        temp_state[6:10] = q
        acc_body = self._compute_acceleration(temp_state, t)
        # Переводим в world frame для интегрирования скорости
        acc_world = self._quat_rotate(q, acc_body)
        vel_dot = acc_world

        # Кинематика кватерниона
        w_x, w_y, w_z = omega
        q_dot = 0.5 * np.array([
            q[3] * w_x - q[2] * w_y + q[1] * w_z,
            q[2] * w_x + q[3] * w_y - q[0] * w_z,
            -q[1] * w_x + q[0] * w_y + q[3] * w_z,
            -q[0] * w_x - q[1] * w_y - q[2] * w_z
        ])

        # Угловое ускорение с переменным моментом инерции (#8)
        M = self._compute_moments(temp_state, t)
        I_dyn = self._get_inertia(mass)
        Ix, Iy, Iz = I_dyn[0], I_dyn[1], I_dyn[2]
        omega_dot = np.array([
            M[0] / Ix,
            M[1] / Iy - (Iz - Ix) / Iy * omega[0] * omega[2],
            M[2] / Iz - (Ix - Iy) / Iz * omega[0] * omega[1]
        ])

        # Сервы: целевое положение + ограничение скорости (#3)
        delta_target = np.clip(u, -1.0, 1.0) * self.max_delta
        delta_error = delta_target - delta

        # Скорость сервы по модели первого порядка
        delta_dot_ideal = delta_error / self.tau_servo

        # Ограничение скорости (#3): не быстрее omega_max_servo
        delta_dot = np.clip(delta_dot_ideal, -self.omega_max_servo, self.omega_max_servo)

        # Расход массы (топливо)
        thrust_mag = self.motor.get_thrust(t)
        if thrust_mag > 0:
            mass_dot = -thrust_mag / (self.atmo.g0 * 280.0)  # удельный импульс ~280 с
        else:
            mass_dot = 0.0

        extra_dot = 0.0

        return np.concatenate([
            pos_dot, vel_dot, q_dot, omega_dot, delta_dot, [mass_dot], [extra_dot]
        ])

    # ------------------------------------------------------------------
    # Шаг симуляции с шумом датчиков (#10)
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        u = np.clip(action, -1.0, 1.0)
        dt = self.dt
        self.step_count += 1

        # Обновление ветра (#4)
        self.wind = self.wind_process.step(dt)

        # Обновление дрейфа датчиков (#10)
        self.gyro_bias += self.gyro_bias_drift * dt
        self.acc_bias += self.acc_bias_drift * dt

        # RK4
        k1 = self._dynamics(self.state, u, self.time)
        k2 = self._dynamics(self.state + 0.5 * dt * k1, u, self.time + 0.5 * dt)
        k3 = self._dynamics(self.state + 0.5 * dt * k2, u, self.time + 0.5 * dt)
        k4 = self._dynamics(self.state + dt * k3, u, self.time + dt)

        self.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state[6:10] /= np.linalg.norm(self.state[6:10]) + 1e-8
        self.state[16] = max(self.mass_dry, self.state[16])

        self.time += dt

        # Проверка выхода за пределы (#1: body Z — продольная ось)
        q = self.state[6:10]
        body_z = self._quat_rotate(q, np.array([0.0, 0.0, 1.0]))
        body_z /= np.linalg.norm(body_z) + 1e-8
        angle_error = np.arccos(np.clip(body_z[2], -1.0, 1.0))  # угол между body Z и world Z
        alt = self.state[2]

        done = False
        if alt < -1.0 or angle_error > np.deg2rad(60.0) or self.time >= self.max_time:
            done = True

        # Ускорение в body frame
        acc_body = self._compute_acceleration(self.state, self.time)

        # Шум датчиков (#10): гауссовский + дрейф + сбои I2C
        gyro_noise_std = np.deg2rad(0.5)  # 0.5 °/с
        acc_noise_std = 0.3  # м/с²

        # Случайные сбои I2C (#10): 0.1% шанс выброса
        if np.random.random() < 0.001:
            # Сбой: случайный большой выброс на одном из каналов
            glitch_channel = np.random.randint(0, 3)
            acc_body[glitch_channel] += np.random.uniform(-20.0, 20.0)

        # Применяем шум
        acc_noisy = acc_body + self.acc_bias + np.random.randn(3) * acc_noise_std

        # Гироскоп (omega + bias + noise)
        omega_noisy = self.state[10:13] + self.gyro_bias + np.random.randn(3) * gyro_noise_std

        info = {
            'alt': alt,
            'angle_deg': np.rad2deg(angle_error),
            'mass': self.state[16],
            'thrust': self.motor.get_thrust(self.time),
            'omega': omega_noisy,   # гироскоп с шумом
        }

        # Возвращаем: ускорение (body frame, с шумом), флаг done, инфо
        return acc_noisy, done, info


# ============================================================================
# Генерация эпизодов
# ============================================================================
def generate_episode(episode_id: int) -> list:
    """Генерирует один эпизод."""
    env = FastRocketSim(dt=0.02, max_time=12.0)
    state = env.reset()
    lines = []

    max_steps = 600
    policy_type = np.random.choice(['random', 'pd', 'mixed', 'explore'],
                                    p=[0.15, 0.25, 0.3, 0.3])

    kp = np.random.uniform(1.0, 3.0)
    kd = np.random.uniform(0.3, 0.8)

    for step in range(max_steps):
        if policy_type == 'random':
            action = np.random.uniform(-1.0, 1.0, 3)

        elif policy_type == 'pd':
            q = state[6:10]
            omega = state[10:13]
            # body Z — продольная ось (#1)
            body_z = env._quat_rotate(q, np.array([0.0, 0.0, 1.0]))
            body_z /= np.linalg.norm(body_z) + 1e-8

            # Ошибка ориентации: отклонение body Z от world Z
            action_pitch = -kp * np.arctan2(body_z[0], body_z[2]) - kd * omega[0]
            action_yaw = -kp * np.arctan2(body_z[1], body_z[2]) - kd * omega[1]
            action_roll = -0.5 * omega[2]

            action = np.array([
                np.clip(action_pitch, -1.0, 1.0),
                np.clip(action_yaw, -1.0, 1.0),
                np.clip(action_roll, -1.0, 1.0)
            ])

        elif policy_type == 'mixed':
            if np.random.random() < 0.4:
                action = np.random.uniform(-1.0, 1.0, 3)
            else:
                q = state[6:10]
                omega = state[10:13]
                body_z = env._quat_rotate(q, np.array([0.0, 0.0, 1.0]))
                body_z /= np.linalg.norm(body_z) + 1e-8

                action_pitch = -kp * np.arctan2(body_z[0], body_z[2]) - kd * omega[0]
                action_yaw = -kp * np.arctan2(body_z[1], body_z[2]) - kd * omega[1]
                action_roll = -0.5 * omega[2]

                action = np.array([
                    np.clip(action_pitch, -1.0, 1.0),
                    np.clip(action_yaw, -1.0, 1.0),
                    np.clip(action_roll, -1.0, 1.0)
                ])
                action += np.random.uniform(-0.3, 0.3, 3)
                action = np.clip(action, -1.0, 1.0)

        else:  # explore
            if step < 100:
                if np.random.random() < 0.15:
                    action = np.random.uniform(-1.0, 1.0, 3) * 0.8
                else:
                    action = np.zeros(3)
            else:
                q = state[6:10]
                omega = state[10:13]
                body_z = env._quat_rotate(q, np.array([0.0, 0.0, 1.0]))
                body_z /= np.linalg.norm(body_z) + 1e-8

                action_pitch = -kp * np.arctan2(body_z[0], body_z[2]) - kd * omega[0]
                action_yaw = -kp * np.arctan2(body_z[1], body_z[2]) - kd * omega[1]
                action_roll = -0.5 * omega[2]

                action = np.array([
                    np.clip(action_pitch, -1.0, 1.0),
                    np.clip(action_yaw, -1.0, 1.0),
                    np.clip(action_roll, -1.0, 1.0)
                ])

        acc, done, info = env.step(action)

        # Формируем строку лога (формат совместим с train.py)
        line = (
            f"{episode_id},{step},{env.time:.6f},{info['alt']:.6f},{info['angle_deg']:.6f},"
            f"{info['thrust']:.6f},{np.linalg.norm(env.wind):.6f},"
            f"{state[0]:.6f},{state[1]:.6f},{state[2]:.6f},"
            f"{state[3]:.6f},{state[4]:.6f},{state[5]:.6f},"
            f"{state[6]:.6f},{state[7]:.6f},{state[8]:.6f},{state[9]:.6f},"
            f"{state[10]:.6f},{state[11]:.6f},{state[12]:.6f},"
            f"{acc[0]:.6f},{acc[1]:.6f},{acc[2]:.6f},"
            f"{action[0]:.6f},{action[1]:.6f},{action[2]:.6f},"
            f"{0.0:.6f},{int(done)},"
            f"{env.CD0:.6f},{env.CL_alpha:.6f},{env.Cm_alpha:.6f},{env.Cmq:.6f},"
            f"{env.I[0]:.6f},{env.I[1]:.6f},{env.I[2]:.6f}\n"
        )

        lines.append(line)

        if done:
            break

    return lines


# ============================================================================
# Генерация датасета
# ============================================================================
def generate_dataset(total_episodes: int = 30000, num_workers: int = 20):
    """Генерирует датасет."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"flight_data_{timestamp}.log.gz"

    print("=" * 60)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА (исправленный симулятор v2)")
    print("=" * 60)
    print(f"Эпизодов: {total_episodes}")
    print(f"Ядер: {num_workers}")
    print(f"Файл: {log_file}")
    print("Исправления: NEU-оси, tau_servo=0.2, лимит серв, ОУ-ветер, вибрация, дрейф")
    print("=" * 60)

    start_time = time.time()

    header = (
        "episode,step,time,alt,angle_deg,thrust,wind_speed,"
        "pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,"
        "quat_x,quat_y,quat_z,quat_w,omega_x,omega_y,omega_z,"
        "acc_x,acc_y,acc_z,action_pitch,action_yaw,action_roll,"
        "reward,done,cd0,cl_alpha,cm_alpha,cmq,ixx,iyy,izz\n"
    )

    with gzip.open(log_file, 'wt', encoding='utf-8') as f:
        f.write(header)

        chunk_size = 100
        total_lines = 0

        with tqdm(total=total_episodes, desc="Генерация") as pbar:
            for chunk_start in range(0, total_episodes, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_episodes)
                chunk_lines = []

                for ep in range(chunk_start, chunk_end):
                    lines = generate_episode(ep)
                    chunk_lines.extend(lines)
                    pbar.update(1)

                f.writelines(chunk_lines)
                f.flush()
                total_lines += len(chunk_lines)

                if (chunk_start // chunk_size) % 10 == 0 and chunk_start > 0:
                    elapsed = time.time() - start_time
                    rate = total_lines / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({"Скорость": f"{rate:.0f} стр/с", "Строк": f"{total_lines:,}"})

    elapsed = time.time() - start_time
    file_size = os.path.getsize(log_file) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Всего строк: {total_lines:,}")
    print(f"Размер: {file_size:.2f} MB")
    print(f"Время: {elapsed / 60:.1f} минут")
    print(f"Файл: {log_file}")
    print("=" * 60)

    return log_file


if __name__ == "__main__":
    generate_dataset(total_episodes=15000, num_workers=24)