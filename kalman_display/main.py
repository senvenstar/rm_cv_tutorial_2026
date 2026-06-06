import cv2 as cv
import numpy as np

# --- 导入必要的模块 ---
# 仿真模块
from sim.camera import Camera
from sim.robot import Robot
from sim.mover import OmniMover

# 瞄准/算法模块
from aim.solver import Solver
from aim.tracker import TrackerByPose

# 工具模块
from utils.plotter import Plotter

# ==============================================================================
# TODO: 可调参数
# ==============================================================================

# 1. 初始状态协方差 P0
# 状态量： [x, vx, y, vy, z, vz,   a,   w, r]
P0_DIAG = [1, 64, 1, 64, 1, 64, 0.4, 100, 1]

# 2. 过程噪声 Q (通过标准差 sigma 定义)
SIGMA_V = 100.0  # 线性运动模型的过程噪声标准差 (m/s^2)
SIGMA_W = 400.0  # 旋转运动模型的过程噪声标准差 (rad/s^2)

# 3. 测量噪声 R
R_DIAG = [0.01, 0.01, 0.01, 0.01]

# # 2. 过程噪声 Q (通过标准差 sigma 定义)
# SIGMA_V = 100.0  # 线性运动模型的过程噪声标准差 (m/s^2)
# SIGMA_W = 400.0  # 旋转运动模型的过程噪声标准差 (rad/s^2)

# # 3. 测量噪声 R
# R_DIAG = [0.001, 0.001, 0.001, 0.001]

# # 2. 过程噪声 Q (通过标准差 sigma 定义)
# SIGMA_V = 1.0  # 线性运动模型的过程噪声标准差 (m/s^2)
# SIGMA_W = 4.0  # 旋转运动模型的过程噪声标准差 (rad/s^2)

# # 3. 测量噪声 R
# R_DIAG = [0.01, 0.01, 0.01, 0.01]

# ==============================================================================
# --- 仿真参数设置 ---
T = 5
dt = 1 / 60
np.random.seed(42)

# --- 初始化仿真环境和对象 ---

# 1. 初始化机器人和移动模型

# TODO: 有3种目标运动场景需要做跟随，你需要每次只解注释其中一个
# ==============================================================================
# a. 原地小陀螺
# ==============================================================================
robot = Robot(z0=0.1, h=0)
mover = OmniMover(
    angular_mode="sine_wave",
    p1=(3, 0),
    p2=(3, 0),
    v_linear=0,
    vel_amplitude=0,
    vel_frequency=np.pi,
    vel_center=3,
)

# ==============================================================================
# b. x轴平移小陀螺
# ==============================================================================
# robot = Robot(z0=0.1, h=0)
# mover = OmniMover(
#     angular_mode="sine_wave",
#     p1=(3, -1),
#     p2=(3, 1),
#     v_linear=1,
#     vel_amplitude=0,
#     vel_frequency=np.pi,
#     vel_center=6,
# )

# ==============================================================================
# c. xy平面移动小陀螺
# ==============================================================================
# robot = Robot(z0=0.1, h=0)
# mover = OmniMover(
#     angular_mode="sine_wave",
#     p1=(2, -1),
#     p2=(4.5, 1.5),
#     v_linear=1.5,
#     vel_amplitude=0,
#     vel_frequency=np.pi,
#     vel_center=9,
# )

# 2. 初始化相机和解算器
cam = Camera()
solver = Solver(cam.cam_mat, cam.dist_coeffs)

# 3. 初始化追踪器 (使用上面的配置参数)
tracker_cv = TrackerByPose(
    r0=robot.r,
    P0_diag=P0_DIAG,
    sigma_v=SIGMA_V,
    sigma_w=SIGMA_W,
    R_diag=R_DIAG,
)

# 4. 初始化绘图工具
plotter = Plotter()

# --- 主仿真循环 ---
print("仿真开始，按 'q' 键退出...")
for t in np.arange(0, T, dt):
    # 1. 移动机器人，获取真实状态
    mover.move(robot, dt)

    # 2. 相机观测，获得带噪声的2D图像点
    armor = robot.observe(cam)
    points = armor.project(cam)
    points += np.random.normal(0, 1.1, points.shape)

    # 3. 解算器解算位姿
    armor_yaw, armor_pos = solver.lsqr(points, cam.rot, cam.pos)

    # 4. 追踪器进行滤波
    tracker_cv.track(t, armor_yaw, armor_pos, armor.i)

    # --- 数据记录 (与完整系统一致的绘图方式) ---
    # 记录真实值 (x, vx, y, vy, yaw, w)
    plotter.add(
        t,
        robot.pos[0],
        mover.v[0],
        robot.pos[1],
        mover.v[1],
        robot.yaw,
        mover.w,
        name=("x_truth", "vx_truth", "y_truth", "vy_truth", "yaw_truth", "w_truth"),
    )

    # 记录滤波后的完整状态向量
    if tracker_cv.ekf is not None:
        plotter.add(
            t,
            *tracker_cv.ekf.x,
            name=("x", "vx", "y", "vy", "z", "vz", "yaw", "w", "r"),
            mode="lines+markers",
        )

    # --- 图像化显示 ---
    img = cam.render(*robot.armors)
    cv.putText(img, f"t={t:.2f}s", (10, 30), 0, 1, (255, 255, 255))
    cv.polylines(img, [points.astype(int)], True, (0, 0, 255), 2)
    cv.imshow("Simulation View", img)
    if cv.waitKey(1) == ord("q"):
        break

# --- 结果可视化 (修改部分) ---
print("仿真结束，显示结果图表...")
# 将所有需要绘图的变量名放入一个元组中，即可在同一张图里显示
plotter.show(
    (
        "x_truth",
        "x",
        "vx_truth",
        "vx",
        "y_truth",
        "y",
        "vy_truth",
        "vy",
        "yaw_truth",
        "yaw",
        "w_truth",
        "w",
    ),
)
cv.destroyAllWindows()
