import cv2 as cv
import numpy as np

from numpy.typing import NDArray
from scipy.optimize import least_squares

from utils.math_utils import *


class Solver:
    def __init__(self, cam_mat: NDArray, dist_coeffs: NDArray):
        self.cam_mat = cam_mat
        self.dist_coeffs = dist_coeffs

    def pnp(
        self, points: NDArray, cam_rot: NDArray, cam_pos: NDArray
    ) -> tuple[float, NDArray]:
        _, rvec, tvec = cv.solvePnP(
            ARMOR_POINTS,
            points,
            self.cam_mat,
            self.dist_coeffs,
            flags=cv.SOLVEPNP_IPPE,
        )

        rot_in_cv = cv.Rodrigues(rvec)[0]
        pos_in_cv = tvec.reshape(-1)

        rot_cv_to_world = rot_world_to_cv(cam_rot).T
        rot_in_world = rot_cv_to_world @ rot_in_cv
        pos_in_world = rot_cv_to_world @ pos_in_cv + cam_pos

        armor_yaw = rot_to_ypr(rot_in_world)[0]
        armor_pos = pos_in_world

        return armor_yaw, armor_pos

    def lsqr(
        self, points: NDArray, cam_rot: NDArray, cam_pos: NDArray
    ) -> tuple[float, NDArray]:
        def f(x: NDArray):
            yaw, *pos = x
            rot = armor_yaw_to_rot(yaw)
            reprojected_points = project(
                ARMOR_POINTS,
                rot,
                pos,
                cam_rot,
                cam_pos,
                self.cam_mat,
                self.dist_coeffs,
            )
            residuals = reprojected_points - points
            return residuals.flatten()

        def jac(x: NDArray):
            yaw, *pos = x
            rot = armor_yaw_to_rot(yaw)
            j_rot_pos_to_proj = jacobian_rot_pos_to_project(
                ARMOR_POINTS,
                rot,
                pos,
                cam_rot,
                cam_pos,
                self.cam_mat,
                self.dist_coeffs,
            )
            j_rot_to_proj = j_rot_pos_to_proj[:, :9]
            j_pos_to_proj = j_rot_pos_to_proj[:, 9:]
            j_yaw_to_proj = j_rot_to_proj @ jacobian_armor_yaw_to_rot(yaw)
            return np.block([j_yaw_to_proj, j_pos_to_proj])

        armor_yaw0 = rot_to_ypr(cam_rot)[0]
        _, armor_pos0 = self.pnp(points, cam_rot, cam_pos)
        x0 = [armor_yaw0, *armor_pos0]

        result = least_squares(f, x0, jac)
        armor_yaw, *armor_pos = result.x

        return armor_yaw, armor_pos
