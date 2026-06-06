import numpy as np

from typing import Callable
from numpy.typing import NDArray


class ExtendedKalmanFilter:
    def __init__(
        self,
        x0: NDArray,
        P0: NDArray,
    ) -> None:
        """
        x0: 初始状态
        P0: 初始状态噪声
        """
        self.x = x0
        self.P = P0

    def copy(self):
        """返回当前EKF对象的一个深拷贝"""
        return ExtendedKalmanFilter(self.x.copy(), self.P.copy())

    def predict(
        self,
        f: Callable[[NDArray], NDArray],
        jacobian_f: Callable[[NDArray], NDArray],
        Q: NDArray,
    ) -> None:
        """
        f: 状态转移函数 f(x) -> x
        jacobian_f: 状态转移函数的雅可比矩阵 jacobian_f(x) -> F
        Q: 过程噪声
        """
        # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb
        F = jacobian_f(self.x)
        self.x = f(self.x)
        self.P = F @ self.P @ F.T + Q

    def update(
        self,
        z: NDArray,
        h: Callable[[NDArray], NDArray],
        jacobian_h: Callable[[NDArray], NDArray],
        R: NDArray,
        x_add: Callable[[NDArray, NDArray], NDArray] = None,
        z_sub: Callable[[NDArray, NDArray], NDArray] = None,
    ) -> None:
        """
        z: 量测向量
        h: 量测函数 h(x) -> z
        jacobian_h: 量测函数的雅可比矩阵 jacobian_h(x) -> H
        R: 量测噪声
        x_add: 定义状态向量加法, 便于处理角度突变
        z_sub: 定义量测向量减法, 便于处理角度突变
        """
        if x_add is None:
            x_add = np.add

        if z_sub is None:
            z_sub = np.subtract

        H = jacobian_h(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)

        y = z_sub(z, h(self.x))
        self.x = x_add(self.x, K @ y)

        # Stable Compution of the Posterior Covariance
        # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
        I = np.identity(self.x.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        nis = y.T @ np.linalg.inv(S) @ y
        return nis


def test_ekf():
    import matplotlib.pyplot as plt

    np.random.seed(42)

    ekf = ExtendedKalmanFilter(np.array([0, 0]), np.diag([10, 10]))

    def f(x):
        return np.array([x[0] + x[1], x[1]])

    def jacobian_f(x):
        return np.array([[1, 1], [0, 1]])

    def h(x):
        return np.array([x[0]])

    def jacobian_h(x):
        return np.array([[1, 0]])

    measured = []
    estimated = []

    for i in range(50):
        z = i + np.random.randn()
        ekf.predict(f, jacobian_f, np.diag([0, 0]))
        ekf.update(z, h, jacobian_h, np.diag([1]))
        measured.append(z)
        estimated.append(ekf.x[0])

    plt.plot(measured, label="measured", marker="x", ls="")
    plt.plot(estimated, label="estimated")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_ekf()
