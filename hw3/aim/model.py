import numpy as np

from numpy.typing import NDArray

from utils.ekf import ExtendedKalmanFilter
from utils.math_utils import *


def x_add(a: NDArray, b: NDArray) -> NDArray:
    c = a + b
    c[6] = limit_pi(c[6])
    return c


def f(x: NDArray, dt: float) -> NDArray:
    x = jacobian_f(x, dt) @ x
    x[6] = limit_pi(x[6])
    return x


def jacobian_f(x: NDArray, dt: float) -> NDArray:
    return np.array(
        [
            [1, dt, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )


def x_to_armor_yaw_pos(x: NDArray, i: int) -> tuple[float, NDArray]:
    x, _, y, _, z, _, yaw, _, r = x
    armor_yaw = limit_pi(yaw + i * np.pi / 2)
    armor_pos = np.array([x, y, z]) - r * np.array([cos(armor_yaw), sin(armor_yaw), 0])
    return armor_yaw, armor_pos


def jacobian_x_to_armor_yaw_pos(x: NDArray, i: int) -> NDArray:
    r = x[8]
    armor_yaw, _ = x_to_armor_yaw_pos(x, i)
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, r * sin(armor_yaw), 0, -cos(armor_yaw)],
            [0, 0, 1, 0, 0, 0, -r * cos(armor_yaw), 0, -sin(armor_yaw)],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )


def init_ekf(
    r0: float, P0_diag: NDArray, armor_yaw: float, armor_pos: NDArray, armor_i: int
) -> ExtendedKalmanFilter:
    yaw = limit_pi(armor_yaw - armor_i * np.pi / 2)
    pos = armor_pos + r0 * np.array([cos(armor_yaw), sin(armor_yaw), 0])

    # x vx y vy z vz yaw w r
    x0 = np.array([pos[0], 0, pos[1], 0, pos[2], 0, yaw, 0, r0])
    P0 = np.diag(P0_diag)

    return ExtendedKalmanFilter(x0, P0)


def predict_ekf(ekf: ExtendedKalmanFilter, sigma_v: float, sigma_w: float, dt: float):
    # Piecewise White Noise Model
    # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
    v = sigma_v**2  # 加速度方差
    w = sigma_w**2  # 角加速度方差
    a = dt**4 / 4
    b = dt**3 / 2
    c = dt**2
    Q = np.array(
        [
            [a * v, b * v, 0, 0, 0, 0, 0, 0, 0],
            [b * v, c * v, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, a * v, b * v, 0, 0, 0, 0, 0],
            [0, 0, b * v, c * v, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, a * v, b * v, 0, 0, 0],
            [0, 0, 0, 0, b * v, c * v, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, a * w, b * w, 0],
            [0, 0, 0, 0, 0, 0, b * w, c * w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    ekf.predict(lambda x: f(x, dt), lambda x: jacobian_f(x, dt), Q)
