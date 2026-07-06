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


# Атмосфера
class Atmosphere:
    def __init__(self):
        self.g0 = 9.80665
        self.rho0 = 1.225
        self.H = 8500.0

    def get_density(self, alt: float) -> float:
        alt = max(0.0, alt)
        return self.rho0 * np.exp(-alt / self.H)


# Двигатель
class Motor:
    def __init__(self, total_impulse: float = 1800.0, burn_time: float = 3.0):
        self.total_impulse = total_impulse
        self.burn_time = burn_time
        self.thrust_peak = total_impulse / burn_time * 1.4

    def get_thrust(self, t: float) -> float:
        if t < 0 or t >= self.burn_time:
            return 0.0
        t_norm = t / self.burn_time
        if t_norm < 0.05:
            return self.thrust_peak * (t_norm / 0.05)
        elif t_norm < 0.6:
            return self.thrust_peak
        else:
            decay = 1.0 - (t_norm - 0.6) / 0.4
            return self.thrust_peak * max(0.0, decay)


# Быстрый симулятор
class FastRocketSim:
    def __init__(self, dt: float = 0.02, max_time: float = 12.0):
        self.dt = dt
        self.max_time = max_time
        self.atmo = Atmosphere()

        self.base_mass_dry = 2.5
        self.base_mass_prop = 1.2
        self.S_ref = 0.008
        self.L_ref = 0.5

        self.base_CD0 = 0.35
        self.base_CL_alpha = 2.5
        self.base_Cm_alpha = -0.5
        self.base_Cmq = -2.0

        self.base_Ixx = 0.01
        self.base_Iyy = 0.25
        self.base_Izz = 0.25

        self.max_delta = np.deg2rad(25.0)
        self.tau_servo = 0.05

        self.state = np.zeros(18)
        self.time = 0.0
        self.wind = np.zeros(3)

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

    def reset(self) -> np.ndarray:
        self.time = 0.0
        self.state = np.zeros(18)
        self.step_count = 0

        mass_factor = np.random.uniform(0.8, 1.2)
        self.mass_dry = self.base_mass_dry * mass_factor
        self.mass_prop = self.base_mass_prop * np.random.uniform(0.85, 1.15)
        self.total_mass = self.mass_dry + self.mass_prop

        self.CD0 = self.base_CD0 * np.random.uniform(0.7, 1.3)
        self.CL_alpha = self.base_CL_alpha * np.random.uniform(0.7, 1.3)
        self.Cm_alpha = self.base_Cm_alpha * np.random.uniform(0.7, 1.3)
        self.Cmq = self.base_Cmq * np.random.uniform(0.7, 1.3)

        I_factor = np.random.uniform(0.8, 1.2)
        self.I = np.array([
            self.base_Ixx * I_factor * np.random.uniform(0.9, 1.1),
            self.base_Iyy * I_factor * np.random.uniform(0.9, 1.1),
            self.base_Izz * I_factor * np.random.uniform(0.9, 1.1)
        ])

        wind_speed = np.random.uniform(0.0, 5.0)
        wind_angle = np.random.uniform(0, 2 * np.pi)
        self.wind = np.array([
            wind_speed * np.cos(wind_angle),
            wind_speed * np.sin(wind_angle),
            0.0
        ])

        pitch_dev = np.deg2rad(np.random.uniform(-15.0, 15.0))
        yaw_dev = np.deg2rad(np.random.uniform(-15.0, 15.0))
        start_alt = np.random.uniform(0.0, 3.0)
        start_vx = np.random.uniform(-3.0, 3.0)
        start_vy = np.random.uniform(-3.0, 3.0)
        start_vz = np.random.uniform(-1.0, 2.0)

        impulse_factor = np.random.uniform(0.85, 1.15)
        burn_time_factor = np.random.uniform(0.9, 1.1)
        self.motor = Motor(
            total_impulse=1800.0 * impulse_factor,
            burn_time=3.0 * burn_time_factor
        )

        self.state[0] = np.random.uniform(-1.0, 1.0)
        self.state[1] = np.random.uniform(-1.0, 1.0)
        self.state[2] = start_alt
        self.state[3] = start_vx
        self.state[4] = start_vy
        self.state[5] = start_vz

        q_init = R.from_euler('Y', -np.pi / 2).as_quat()
        q_pitch = R.from_euler('Y', pitch_dev).as_quat()
        q_yaw = R.from_euler('Z', yaw_dev).as_quat()
        roll_dev = np.deg2rad(np.random.uniform(-5.0, 5.0))
        q_roll = R.from_euler('X', roll_dev).as_quat()

        q_dev = self._quat_multiply(q_pitch, q_yaw)
        q_dev = self._quat_multiply(q_dev, q_roll)
        self.state[6:10] = self._quat_multiply(q_init, q_dev)
        self.state[6:10] /= np.linalg.norm(self.state[6:10])

        self.state[16] = self.total_mass
        self.state[17] = 0.0
        self.state[13:16] = np.random.uniform(-0.1, 0.1, 3) * self.max_delta

        return self.state.copy()

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

    def _compute_acceleration(self, state: np.ndarray, t: float) -> np.ndarray:
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        mass = max(self.mass_dry, state[16])

        alt = max(0.0, pos[2])
        rho = self.atmo.get_density(alt)
        v_rel = vel - self.wind
        V = np.linalg.norm(v_rel)

        F_gravity = np.array([0.0, 0.0, -mass * self.atmo.g0])

        F_aero = np.zeros(3)
        if V > 0.01:
            v_body = self._quat_rotate_inv(q, v_rel)
            alpha = np.arctan2(v_body[2], v_body[0])
            beta = np.arctan2(v_body[1], v_body[0])
            q_bar = 0.5 * rho * V ** 2

            v_dir = v_rel / V
            F_drag = -q_bar * self.S_ref * self.CD0 * v_dir

            lift_body = np.array([0.0, 0.0, q_bar * self.S_ref * self.CL_alpha * alpha])
            F_lift = self._quat_rotate(q, lift_body)

            side_body = np.array([0.0, q_bar * self.S_ref * self.CL_alpha * beta, 0.0])
            F_side = self._quat_rotate(q, side_body)

            F_aero = F_drag + F_lift + F_side

        F_thrust = np.zeros(3)
        thrust_mag = self.motor.get_thrust(t)
        if thrust_mag > 0:
            body_x = self._quat_rotate(q, np.array([1.0, 0.0, 0.0]))
            body_x = body_x / (np.linalg.norm(body_x) + 1e-8)
            F_thrust = thrust_mag * body_x

        F_total = F_gravity + F_aero + F_thrust
        return F_total / mass

    def _compute_moments(self, state: np.ndarray, t: float) -> np.ndarray:
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        delta = state[13:16]

        alt = max(0.0, pos[2])
        rho = self.atmo.get_density(alt)
        v_rel = vel - self.wind
        V = np.linalg.norm(v_rel)
        q_bar = 0.5 * rho * V ** 2 if V > 0.01 else 0.0

        M_aero = np.zeros(3)
        if V > 0.01:
            v_body = self._quat_rotate_inv(q, v_rel)
            alpha = np.arctan2(v_body[2], v_body[0])
            beta = np.arctan2(v_body[1], v_body[0])

            M_restore = q_bar * self.S_ref * self.L_ref * self.Cm_alpha * np.array([
                0.0,
                alpha,
                beta
            ])

            M_damp = q_bar * self.S_ref * self.L_ref ** 2 * self.Cmq / (V + 0.01) * omega

            M_control = q_bar * self.S_ref * self.L_ref * 0.1 * np.array([
                0.0,
                delta[0],
                delta[1]
            ])

            M_aero = M_restore + M_damp + M_control

        M_thrust = np.zeros(3)
        thrust_mag = self.motor.get_thrust(t)
        if thrust_mag > 0:
            r_nozzle = np.array([-0.4, 0.0, 0.0])
            thrust_dir = np.array([1.0, 0.0, 0.0])
            thrust_dir += np.array([
                0.0,
                np.deg2rad(np.random.uniform(-0.5, 0.5)),
                np.deg2rad(np.random.uniform(-0.5, 0.5))
            ])
            thrust_dir /= np.linalg.norm(thrust_dir)
            F_thrust = thrust_mag * thrust_dir
            M_thrust = np.cross(r_nozzle, F_thrust)

        return M_aero + M_thrust

    def _dynamics(self, state: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        delta = state[13:16]
        mass = max(self.mass_dry, state[16])

        q = q / (np.linalg.norm(q) + 1e-8)

        pos_dot = vel

        temp_state = state.copy()
        temp_state[6:10] = q
        acc = self._compute_acceleration(temp_state, t)
        vel_dot = acc

        w_x, w_y, w_z = omega
        q_dot = 0.5 * np.array([
            q[3] * w_x - q[2] * w_y + q[1] * w_z,
            q[2] * w_x + q[3] * w_y - q[0] * w_z,
            -q[1] * w_x + q[0] * w_y + q[3] * w_z,
            -q[0] * w_x - q[1] * w_y - q[2] * w_z
        ])

        M = self._compute_moments(temp_state, t)
        I = self.I
        omega_dot = np.array([
            M[0] / I[0],
            M[1] / I[1] - (I[2] - I[0]) / I[1] * omega[0] * omega[2],
            M[2] / I[2] - (I[0] - I[1]) / I[2] * omega[0] * omega[1]
        ])

        delta_target = np.clip(u, -1.0, 1.0) * self.max_delta
        delta_dot = (delta_target - delta) / self.tau_servo

        thrust_mag = self.motor.get_thrust(t)
        mass_dot = -thrust_mag / (self.atmo.g0 * 280.0)

        extra_dot = 0.0

        return np.concatenate([
            pos_dot, vel_dot, q_dot, omega_dot, delta_dot, [mass_dot], [extra_dot]
        ])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        u = np.clip(action, -1.0, 1.0)
        dt = self.dt
        self.step_count += 1

        k1 = self._dynamics(self.state, u, self.time)
        k2 = self._dynamics(self.state + 0.5 * dt * k1, u, self.time + 0.5 * dt)
        k3 = self._dynamics(self.state + 0.5 * dt * k2, u, self.time + 0.5 * dt)
        k4 = self._dynamics(self.state + dt * k3, u, self.time + dt)

        self.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state[6:10] /= np.linalg.norm(self.state[6:10]) + 1e-8
        self.state[16] = max(self.mass_dry, self.state[16])

        self.time += dt

        q = self.state[6:10]
        up_body = self._quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        up_body /= np.linalg.norm(up_body) + 1e-8
        angle_error = np.arccos(np.clip(up_body[2], -1.0, 1.0))
        alt = self.state[2]

        done = False
        if alt < -1.0 or angle_error > np.deg2rad(60.0) or self.time >= self.max_time:
            done = True

        acc = self._compute_acceleration(self.state, self.time)

        info = {
            'alt': alt,
            'angle_deg': np.rad2deg(angle_error),
            'mass': self.state[16],
            'thrust': self.motor.get_thrust(self.time)
        }

        return acc, done, info


def generate_episode(episode_id: int) -> list:
    """Генерирует один эпизод"""
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
            up_body = env._quat_rotate(q, np.array([1.0, 0.0, 0.0]))
            up_body /= np.linalg.norm(up_body) + 1e-8

            action_pitch = -kp * np.arctan2(up_body[0], up_body[2]) - kd * omega[1]
            action_yaw = -kp * np.arctan2(up_body[1], up_body[2]) - kd * omega[2]
            action_roll = -0.5 * omega[0]

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
                up_body = env._quat_rotate(q, np.array([1.0, 0.0, 0.0]))
                up_body /= np.linalg.norm(up_body) + 1e-8

                action_pitch = -kp * np.arctan2(up_body[0], up_body[2]) - kd * omega[1]
                action_yaw = -kp * np.arctan2(up_body[1], up_body[2]) - kd * omega[2]
                action_roll = -0.5 * omega[0]

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
                up_body = env._quat_rotate(q, np.array([1.0, 0.0, 0.0]))
                up_body /= np.linalg.norm(up_body) + 1e-8

                action_pitch = -kp * np.arctan2(up_body[0], up_body[2]) - kd * omega[1]
                action_yaw = -kp * np.arctan2(up_body[1], up_body[2]) - kd * omega[2]
                action_roll = -0.5 * omega[0]

                action = np.array([
                    np.clip(action_pitch, -1.0, 1.0),
                    np.clip(action_yaw, -1.0, 1.0),
                    np.clip(action_roll, -1.0, 1.0)
                ])

        acc, done, info = env.step(action)

        # Формируем строку лога
        line = f"{episode_id},{step},{env.time:.6f},{info['alt']:.6f},{info['angle_deg']:.6f},{info['thrust']:.6f},{np.linalg.norm(env.wind):.6f},{state[0]:.6f},{state[1]:.6f},{state[2]:.6f},{state[3]:.6f},{state[4]:.6f},{state[5]:.6f},{state[6]:.6f},{state[7]:.6f},{state[8]:.6f},{state[9]:.6f},{state[10]:.6f},{state[11]:.6f},{state[12]:.6f},{acc[0]:.6f},{acc[1]:.6f},{acc[2]:.6f},{action[0]:.6f},{action[1]:.6f},{action[2]:.6f},{0.0:.6f},{int(done)},{env.CD0:.6f},{env.CL_alpha:.6f},{env.Cm_alpha:.6f},{env.Cmq:.6f},{env.I[0]:.6f},{env.I[1]:.6f},{env.I[2]:.6f}\n"

        lines.append(line)

        if done:
            break

    return lines


def generate_dataset(total_episodes: int = 30000, num_workers: int = 20):
    """Генерирует датасет"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"flight_data_{timestamp}.log.gz"

    print("=" * 60)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА")
    print("=" * 60)
    print(f"Эпизодов: {total_episodes}")
    print(f"Ядер: {num_workers}")
    print(f"Файл: {log_file}")
    print("=" * 60)

    start_time = time.time()

    header = "episode,step,time,alt,angle_deg,thrust,wind_speed," + \
             "pos_x,pos_y,pos_z,vel_x,vel_y,vel_z," + \
             "quat_x,quat_y,quat_z,quat_w,omega_x,omega_y,omega_z," + \
             "acc_x,acc_y,acc_z,action_pitch,action_yaw,action_roll," + \
             "reward,done,cd0,cl_alpha,cm_alpha,cmq,ixx,iyy,izz\n"

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

    generate_dataset(total_episodes=10000, num_workers=20)