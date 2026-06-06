import numpy as np

from numpy.typing import NDArray, ArrayLike

from aim.model import *
from utils.ekf import ExtendedKalmanFilter
from utils.math_utils import *


JUMP_THRESH = np.deg2rad(45)


def z_sub(a: NDArray, b: NDArray) -> NDArray:
    c = a - b
    c[0] = limit_pi(c[0])
    c[1] = limit_pi(c[1])
    return c


def h(x: NDArray, i: int) -> NDArray:
    armor_yaw, armor_pos = x_to_armor_yaw_pos(x, i)
    return np.array([armor_yaw, *pos_to_aed(armor_pos)])


def jacobian_f(x: NDArray, i: int) -> NDArray:
    _, armor_pos = x_to_armor_yaw_pos(x, i)
    jacobian_yaw_pos_to_yaw_aed = np.block(
        [
            [1, np.zeros((1, 3))],
            [np.zeros((3, 1)), jacobian_pos_to_aed(armor_pos)],
        ]
    )
    return jacobian_yaw_pos_to_yaw_aed @ jacobian_x_to_armor_yaw_pos(x, i)


def update_ekf(
    ekf: ExtendedKalmanFilter,
    R_diag: NDArray,
    armor_yaw: float,
    armor_pos: NDArray,
    armor_i: int,
):
    # armor_yaw armor_azim armor_elev armor_dist
    z = np.array([armor_yaw, *pos_to_aed(armor_pos)])
    R = np.diag(R_diag)

    ekf.update(
        z,
        lambda x: h(x, armor_i),
        lambda x: jacobian_f(x, armor_i),
        R,
        x_add,
        z_sub,
    )


class TrackerByPose:
    def __init__(
        self,
        r0: float,
        P0_diag: ArrayLike,
        sigma_v: float,
        sigma_w: float,
        R_diag: ArrayLike,
    ):
        self.ekf: ExtendedKalmanFilter = None
        self.t: float = None
        self.armor_i: int = None
        self.armor_yaw: float = None
        self.x_prior: NDArray = None
        self.r0 = r0
        self.P0_diag = P0_diag
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.R_diag = R_diag

    @property
    def predicts(
        self,
    ) -> tuple[None, None, None, None] | tuple[float, float, float, float]:
        if self.x_prior is None:
            return None, None, None, None

        x, _, y, _, z, _, yaw, _, _ = self.x_prior
        self.x_prior = None
        return x, y, z, yaw

    def track(self, t: float, armor_yaw: float, armor_pos: NDArray, armor_i: int):
        if self.ekf is None:
            self.t = t
            self.armor_i = armor_i
            self.armor_yaw = armor_yaw
            self.x_prior = None
            self.ekf = init_ekf(self.r0, self.P0_diag, armor_yaw, armor_pos, armor_i)
            return

        dt = t - self.t
        self.t = t

        predict_ekf(self.ekf, self.sigma_v, self.sigma_w, dt)
        self.x_prior = self.ekf.x.copy()

        dyaw = limit_pi(armor_yaw - self.armor_yaw)
        if dyaw > JUMP_THRESH:
            self.armor_i = (self.armor_i + 1) % 4
            print(f"Jump right to {self.armor_i}")
        elif dyaw < -JUMP_THRESH:
            self.armor_i = (self.armor_i - 1) % 4
            print(f"Jump left to {self.armor_i}")

        update_ekf(self.ekf, self.R_diag, armor_yaw, armor_pos, self.armor_i)
        self.armor_yaw = armor_yaw
