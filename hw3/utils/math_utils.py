import cv2 as cv
import numpy as np

from numpy import sin, cos
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


LIGHT_H = 56e-3
ARMOR_W = 135e-3
ARMOR_POINTS = np.array(
    [
        [0, ARMOR_W / 2, LIGHT_H / 2],
        [0, ARMOR_W / 2, -LIGHT_H / 2],
        [0, -ARMOR_W / 2, -LIGHT_H / 2],
        [0, -ARMOR_W / 2, LIGHT_H / 2],
    ]
)

ARMOR_PITCH = np.deg2rad(15)

ROT_CAM_TO_CV = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ]
)


def rot_world_to_cv(cam_rot: NDArray) -> NDArray:
    return ROT_CAM_TO_CV @ cam_rot.T


def limit_pi(angle: float) -> float:
    """(-pi, pi]"""
    while angle <= -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle


def rot_x(theta: float) -> NDArray:
    return np.array(
        [
            [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)],
        ]
    )


def rot_y(theta: float) -> NDArray:
    return np.array(
        [
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)],
        ]
    )


def rot_z(theta: float) -> NDArray:
    return np.array(
        [
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1],
        ]
    )


def rot_to_ypr(rot: NDArray) -> NDArray:
    """yaw, pitch, roll"""
    return Rotation.from_matrix(rot).as_euler("ZYX", degrees=False)


def pos_to_aed(pos: NDArray) -> NDArray:
    """azimuth, elevation, distance"""
    x, y, z = pos
    azim = np.arctan2(y, x)
    elev = np.arctan2(z, np.hypot(x, y))
    dist = np.sqrt(x**2 + y**2 + z**2)
    return np.array([azim, elev, dist])


def jacobian_pos_to_aed(pos: NDArray) -> NDArray:
    x, y, z = pos
    a = x**2 + y**2
    b = np.sqrt(a)
    c = x**2 + y**2 + z**2
    d = np.sqrt(c)
    return np.array(
        [
            [-y / a, x / a, 0],
            [-x * z / (b * c), -y * z / (b * c), b / c],
            [x / d, y / d, z / d],
        ]
    )


def project(
    points: NDArray,
    rot: NDArray,
    pos: NDArray,
    cam_rot: NDArray,
    cam_pos: NDArray,
    cam_mat: NDArray,
    dist_coeffs: NDArray,
) -> NDArray:
    """
    world frame to pixel frame
    """
    rot_world_to_cv_ = rot_world_to_cv(cam_rot)
    rot_in_cv = rot_world_to_cv_ @ rot
    pos_in_cv = rot_world_to_cv_ @ (pos - cam_pos)

    projected_points = cv.projectPoints(
        points, rot_in_cv, pos_in_cv, cam_mat, dist_coeffs
    )[0].reshape(-1, 2)

    return projected_points


def jacobian_rot_pos_to_project(
    points: NDArray,
    rot: NDArray,
    pos: NDArray,
    cam_rot: NDArray,
    cam_pos: NDArray,
    cam_mat: NDArray,
    dist_coeffs: NDArray,
) -> NDArray:
    rot_world_to_cv_ = rot_world_to_cv(cam_rot)
    rot_in_cv = rot_world_to_cv_ @ rot
    tvec = rot_world_to_cv_ @ (pos - cam_pos)

    jacobian = cv.projectPoints(points, rot_in_cv, tvec, cam_mat, dist_coeffs)[1]

    j_rvec_to_proj = jacobian[:, :3]
    j_tvec_to_proj = jacobian[:, 3:6]

    j_rot_in_cv_to_rvec = cv.Rodrigues(rot_in_cv)[1].T
    j_rot_to_rvec = (
        rot_world_to_cv_.T @ j_rot_in_cv_to_rvec.reshape(-1, 3, 3)
    ).reshape(-1, 9)

    j_rot_to_proj = j_rvec_to_proj @ j_rot_to_rvec
    j_pos_to_proj = j_tvec_to_proj @ rot_world_to_cv_

    return np.block([j_rot_to_proj, j_pos_to_proj])


def armor_yaw_to_rot(yaw: float) -> NDArray:
    return rot_z(yaw) @ rot_y(ARMOR_PITCH)


def jacobian_armor_yaw_to_rot(yaw: float) -> NDArray:
    return np.array(
        [
            -sin(yaw) * cos(ARMOR_PITCH),
            -cos(yaw),
            -sin(ARMOR_PITCH) * sin(yaw),
            cos(ARMOR_PITCH) * cos(yaw),
            -sin(yaw),
            sin(ARMOR_PITCH) * cos(yaw),
            0,
            0,
            0,
        ]
    ).reshape(-1, 1)
