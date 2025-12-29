from typing import List, Callable, Tuple

import numpy as np
import numpy.linalg
import pyray
import time

# --- Optimization Engine ---

def levenberg_marquardt(f, J_f, x0, lamb=1e-2, multiplier=2.0, max_iter=10, tol=1e-4):
    """
    Levenberg-Marquardt optimization to minimize ||f(x)||^2.
    """
    x = np.array(x0, dtype=float)
    n_params = len(x)

    residuals = f(x)
    J = J_f(x)
    current_error = 0.5 * np.sum(residuals ** 2)

    for i in range(max_iter):
        JT = J.T
        H_approx = JT @ J
        gradient = JT @ residuals

        A = H_approx + lamb * np.eye(n_params)
        b = -gradient

        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            lamb *= multiplier
            continue

        x_new = x + delta
        residuals_new = f(x_new)
        new_error = 0.5 * np.sum(residuals_new ** 2)

        if new_error < current_error:
            x = x_new
            lamb /= multiplier
            if np.linalg.norm(delta) < tol:
                break
            current_error = new_error
            residuals = residuals_new
            J = J_f(x)
        else:
            lamb *= multiplier

    return x


class Arm:
    def __init__(self, angles: np.ndarray, lengths: np.ndarray):
        assert angles.shape == lengths.shape
        self.angles = angles
        self.lengths = lengths
    def with_angles(self, angles: np.ndarray):
        return Arm(angles, self.lengths)
    def n(self):
        return self.angles.shape[0]


# TODO make axis variable. hardcode axis of rotation to z_axis
z_axis = np.array([0.0, 0.0, 1.0])

def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3, 3) + np.sin(angle) * K + (1-np.cos(angle))* K@K

def w_i(arm: Arm, i: int) -> np.ndarray:
    w = z_axis
    for j in range(i-1, -1, -1):
        w = rot(z_axis, arm.angles[i]) @ w
    return w

def jacobian(arm: Arm) -> np.ndarray:
    J = np.zeros(shape=(3, arm.n()))
    for i in range(arm.n()):
        J[:, i] = np.cross(w_i(arm, i), (s_i(arm, arm.n()) - s_i(arm, i)))
    return J

def s_i(arm: Arm, i: int) -> np.ndarray:
    tip = np.array([0., 0., 0.])
    for j in range(i-1, -1, -1):
        r = np.array([arm.lengths[j], 0.0, 0.0])
        tip = rot(z_axis, arm.angles[j]) @ (tip + r)

    return tip

def np_to_pyray(vec: np.ndarray) -> pyray.Vector3:
    return pyray.Vector3(vec[0], vec[1], vec[2])

def draw_arm(arm: Arm):
    joint_color = pyray.MAROON
    segment_color = pyray.DARKGRAY
    segment_radius = 0.1
    joint_radius = 0.2

    for i in range(arm.n()):
        current_pos = np_to_pyray(s_i(arm, i))
        pyray.draw_sphere(current_pos, joint_radius, joint_color)

        # Draw link to next joint
        next_pos = np_to_pyray(s_i(arm, i + 1))
        pyray.draw_cylinder_ex(current_pos, next_pos, segment_radius, segment_radius, 16, segment_color)

    # Draw end effector tip
    pyray.draw_sphere(np_to_pyray(s_i(arm, arm.n())), joint_radius * 0.8, pyray.GOLD)


# -------------------------

def arm_3d():
    pyray.init_window(1200, 800, "IK with Levenberg-Marquardt (Fixed)")
    pyray.set_target_fps(60)

    # Camera setup
    camera = pyray.Camera3D()
    camera.position = pyray.Vector3(8.0, 8.0, 8.0)
    camera.target = pyray.Vector3(0.0, 2.0, 0.0)
    camera.up = pyray.Vector3(0.0, 1.0, 0.0)
    camera.fovy = 45.0
    camera.projection = pyray.CameraProjection.CAMERA_PERSPECTIVE

    # Arm Setup
    arm_angles = np.array([0.5, 0.5, 0.5])
    arm_lengths = np.array([3.0, 2.0, 2.0])

    # Define axes: Y-axis (base rotation), Z-axis (shoulder), Z-axis (elbow)
    # This configuration allows full 3D reach.

    arm = Arm(arm_angles, arm_lengths)

    target_pos = np.array([-3.0, -1.0, 0.0])

    while not pyray.window_should_close():
        # --- Update ---
        pyray.update_camera(camera, pyray.CameraMode.CAMERA_ORBITAL)

        # 2. Inverse Kinematics Step
        def residuals_func(x):
            return s_i(arm.with_angles(x), arm.n()) - target_pos

        def jacobian_func(x):
            return jacobian(arm.with_angles(x))

        new_angles = levenberg_marquardt(
            f=residuals_func,
            J_f=jacobian_func,
            x0=arm.angles,
            lamb=1e-2,
            max_iter=5,
            tol=1e-3
        )
        arm.angles = new_angles

        # --- 3. Draw ---
        pyray.begin_drawing()
        pyray.clear_background(pyray.RAYWHITE)
        pyray.begin_mode_3d(camera)

        pyray.draw_grid(20, 1.0)
        pyray.draw_sphere(np_to_pyray(target_pos), 0.3, pyray.GREEN)
        draw_arm(arm)

        pyray.end_mode_3d()

        # UI
        pyray.draw_text("Target Control: W/A/S/D (Planar), Q/E (Height)", 10, 10, 20, pyray.DARKGRAY)
        pyray.draw_text(f"Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]", 10, 40, 20, pyray.BLACK)
        pyray.draw_fps(10, 70)

        pyray.end_drawing()

    pyray.close_window()

if __name__ == "__main__":
    arm_3d()
