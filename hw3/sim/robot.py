import cv2 as cv
import numpy as np

from cv2.typing import MatLike
from numpy.typing import NDArray

from sim.camera import Visible, Camera
from utils.math_utils import *


class Armor(Visible):
    def __init__(self, pos, yaw, i):
        self.pos = pos
        self.yaw = yaw
        self.i = i

    @property
    def rot(self) -> NDArray:
        return armor_yaw_to_rot(self.yaw)

    def project(self, cam: Camera) -> NDArray:
        return project(
            ARMOR_POINTS,
            self.rot,
            self.pos,
            cam.rot,
            cam.pos,
            cam.cam_mat,
            cam.dist_coeffs,
        )

    def draw(self, img: MatLike, cam: Camera):
        points = self.project(cam).astype(int)
        cv.polylines(img, [points], True, (127, 127, 127))


class Robot:
    def __init__(self, x0=3, y0=0, z0=0, r=0.2, h=0.1):
        self.pos = np.array([x0, y0, z0], dtype=float)
        self.yaw = 0
        self.r = r
        self.h = h

    @property
    def armors(self) -> list[Armor]:
        armors = []
        for i in range(4):
            armor_yaw = limit_pi(self.yaw + i * np.pi / 2)
            armor_pos = (
                self.pos
                + np.array([0, 0, self.h])
                - np.array([cos(armor_yaw), sin(armor_yaw), 0]) * self.r
            )
            armors.append(Armor(armor_pos, armor_yaw, i))
        return armors

    def observe(
        self,
        cam: Camera,
    ) -> Armor:
        return min(self.armors, key=lambda a: np.linalg.norm(cam.pos - a.pos))
