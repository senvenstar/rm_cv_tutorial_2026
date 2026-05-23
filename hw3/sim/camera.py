import numpy as np

from typing import Protocol
from cv2.typing import MatLike
from numpy.typing import ArrayLike

from utils.math_utils import *


HK_CS016_6MM_FOV = 45.0
HK_CS016_8MM_FOV = 34.5
HK_CS016_12MM_FOV = 23.4


class Visible(Protocol):
    def draw(self, img: MatLike, cam: "Camera"): ...


class Camera:
    def __init__(
        self,
        x0=0.0,
        y0=0.0,
        z0=0.4,
        w=1440,
        h=1080,
        fov=HK_CS016_6MM_FOV,
        cam_mat: ArrayLike = None,
        dist_coeffs: ArrayLike = None,
    ):
        """
        Args:
            x0 (float): x coordinate of the camera in world coordinates.
            y0 (float): y coordinate of the camera in world coordinates.
            z0 (float): z coordinate of the camera in world coordinates.
            w (int): width of the camera image in pixels.
            h (int): height of the camera image in pixels.
            fov (float): horizontal field of view of the camera in degrees.
            cam_mat (ArrayLike): intrinsic camera matrix. If None, it will be calculated from fov.
            dist_coeffs (ArrayLike): distortion coefficients. If None, it will be set to zero.
        """
        if cam_mat is None:
            f = w / (2 * np.tan(np.deg2rad(fov / 2)))
            cam_mat = np.array(
                [
                    [f, 0, w / 2],
                    [0, f, h / 2],
                    [0, 0, 1],
                ],
            )
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)

        self.pos = np.array([x0, y0, z0])
        self.rot = np.eye(3)
        self.w = w
        self.h = h
        self.cam_mat = np.array(cam_mat)
        self.dist_coeffs = np.array(dist_coeffs)

    def render(self, *objects: Visible) -> MatLike:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for obj in objects:
            obj.draw(img, self)
        return img
