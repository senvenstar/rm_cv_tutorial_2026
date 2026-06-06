import numpy as np

from sim.robot import Robot
from utils.math_utils import *


class OmniMover:
    def __init__(
        self,
        max_linear_accel=3.0,
        angular_mode: str = "variable_speed",
        p1=(3, 1),
        p2=(3, -1),
        v_linear=2.0,
        vel_amplitude=5.0,
        vel_frequency=np.pi,
        vel_phase=0,
        vel_center=6.0,
        w_min=6.0,
        w_max=12.0,
        alpha_accel=10.0,
        alpha_decel=18.0,
        direction=1,
    ):
        self.max_linear_accel = max_linear_accel
        self.angular_mode = angular_mode
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        if np.linalg.norm(self.p2 - self.p1) > 1e-6:
            self.d = (self.p2 - self.p1) / np.linalg.norm(self.p2 - self.p1)
        else:
            self.d = np.array([0.0, 0.0])

        self.target_v = np.array([self.d[0], self.d[1], 0.0]) * v_linear
        self.v = np.array([0.0, 0.0, 0.0])
        self.placed = False

        if self.angular_mode == "sine_wave":
            self._init_sine_wave(vel_amplitude, vel_frequency, vel_phase, vel_center)
        elif self.angular_mode == "variable_speed":
            self._init_variable_speed(w_min, w_max, alpha_accel, alpha_decel, direction)
        else:
            raise ValueError(f"Unsupported angular mode: '{self.angular_mode}'")

    def _init_sine_wave(self, amp, freq, phase, center):
        self._sw_vel_amplitude = amp
        self._sw_vel_frequency = freq
        self._sw_vel_phase = phase
        self._sw_vel_center = center
        self._sw_t = 0.0
        self.w = self._sw_vel_center + self._sw_vel_amplitude * np.sin(
            self._sw_vel_phase
        )
        self.alpha = (
            self._sw_vel_amplitude * self._sw_vel_frequency * np.cos(self._sw_vel_phase)
        )

    def _init_variable_speed(self, w_min, w_max, accel, decel, direction):
        if w_min > w_max:
            raise ValueError("w_min must not be larger than w_max")
        if accel <= 0 or decel <= 0:
            raise ValueError("acc and dec must be positive")
        self._vs_w_min = w_min
        self._vs_w_max = w_max
        self._vs_alpha_accel = accel
        self._vs_alpha_decel = decel
        self._vs_direction = direction
        self.w = self._vs_w_min
        self._vs_state = "accelerating"
        self.alpha = self._vs_alpha_accel

    def move(self, robot: Robot, dt):
        if not self.placed:
            robot.pos[:2] = self.p1
            self.placed = True
            return

        robot_pos_2d = robot.pos[:2]
        goal = self.p2 if (self.target_v[:2] @ self.d) > 0 else self.p1
        if ((goal - robot_pos_2d) @ self.target_v[:2]) < 0.1:
            self.target_v = -self.target_v

        error_v = self.target_v - self.v
        accel_v = error_v / dt
        accel_v_norm = np.linalg.norm(accel_v)
        if accel_v_norm > self.max_linear_accel:
            accel_v = accel_v * (self.max_linear_accel / accel_v_norm)
        self.v += accel_v * dt
        robot.pos += self.v * dt

        if self.angular_mode == "sine_wave":
            self._sw_t += dt
            omega_t_phi = self._sw_vel_frequency * self._sw_t + self._sw_vel_phase
            self.w = self._sw_vel_center + self._sw_vel_amplitude * np.sin(omega_t_phi)
            self.alpha = (
                self._sw_vel_amplitude * self._sw_vel_frequency * np.cos(omega_t_phi)
            )
            robot.yaw = limit_pi(robot.yaw + self.w * dt)
        elif self.angular_mode == "variable_speed":
            if self._vs_state == "accelerating":
                self.alpha = self._vs_alpha_accel
                self.w += self.alpha * dt
                if self.w >= self._vs_w_max:
                    self.w = self._vs_w_max
                    self._vs_state = "decelerating"
            elif self._vs_state == "decelerating":
                self.alpha = -self._vs_alpha_decel
                self.w -= self._vs_alpha_decel * dt
                if self.w <= self._vs_w_min:
                    self.w = self._vs_w_min
                    self._vs_state = "accelerating"
            robot.yaw = limit_pi(robot.yaw + self._vs_direction * self.w * dt)
