from typing import List, Callable

import numpy as np
import numpy.linalg
import pyray
import scipy.optimize

from utils import numerical_jacobian
from good import good, make_deflation_funcs
import time
from icecream import ic


def levenberg_marquardt(f, J_f, x0, lamb=1e-3, multiplier=10, max_iter=100, tol=1e-6):
    """
    Basic implementation of Levenberg-Marquardt optimization.

    Parameters:
    - f: Function returning the residual vector (y_pred - y_true).
    - J_f: Function returning the Jacobian matrix of f.
    - x0: Initial guess for parameters (list or numpy array).
    - lamb: Initial damping factor (lambda).
    - multiplier: Factor to increase/decrease lambda.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence (step size).

    Returns:
    - x: Optimized parameters.
    """
    x = np.array(x0, dtype=float)
    n_params = len(x)

    for i in range(max_iter):
        # 1. Compute residuals and Jacobian at current x
        residuals = f(x)
        J = J_f(x)

        # 2. Compute current error (Sum of Squared Errors)
        current_error = 0.5 * np.sum(residuals ** 2)

        # 3. Construct the Linear System: (J^T * J + lambda * I) * delta = -J^T * residuals
        JT = J.T
        H_approx = JT @ J  # Approximate Hessian
        gradient = JT @ residuals

        # Damped Hessian
        A = H_approx + lamb * np.eye(n_params)
        b = -gradient

        # 4. Solve for delta
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Increasing damping.")
            lamb *= multiplier
            continue

        # 5. Evaluate the candidate point
        x_new = x + delta
        residuals_new = f(x_new)
        new_error = 0.5 * np.sum(residuals_new ** 2)

        # 6. Update logic
        if new_error < current_error:
            # Success: Accept the step and reduce damping (closer to Gauss-Newton)
            x = x_new
            lamb /= multiplier

            # Check for convergence
            if np.linalg.norm(delta) < tol:
                print(f"Converged after {i} iterations.")
                break
        else:
            # Failure: Reject the step and increase damping (closer to Gradient Descent)
            # We do NOT update x here; we try again with a higher lambda
            lamb *= multiplier

    return x


class Arm:
    def __init__(self, angles: np.ndarray, lengths: np.ndarray):
        assert angles.shape == lengths.shape
        self.angles = angles
        self.lengths = lengths
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
        J[:, i] = np.cross(w_i(arm, i), s_i(arm, i))
    return J

def s_i(arm: Arm, i: int) -> np.ndarray:
    tip = np.array([0,0,0])
    for j in range(i, 0, -1):
        r = np.array([arm.lengths[j], 0.0, 0.0])
        tip = rot(z_axis, arm.angles[j]) @ (tip + r)

    return tip

def np_to_pyray(vec: np.ndarray) -> pyray.Vector3:
    return pyray.Vector3(vec[0], vec[1], vec[2])

def draw_arm(arm: Arm):
    # Colors for styling
    joint_color = pyray.MAROON
    segment_color = pyray.DARKGRAY
    segment_radius = 0.1
    joint_radius = 0.2

    for i in range(arm.n()):
        current_pos = np_to_pyray(s_i(arm, i))
        pyray.draw_sphere(current_pos, joint_radius, joint_color)
        if i < arm.n() - 1:
            next_pos = np_to_pyray(s_i(arm,i + 1))
            pyray.draw_cylinder_ex(current_pos,
                                   next_pos,
                                   segment_radius,
                                   segment_radius,
                                   16,
                                   segment_color)


# -------------------------

def arm_3d():
    # 1. Initialize the Window
    pyray.init_window(800, 600, "Raylib Python - Simple 3D Scene")
    pyray.set_target_fps(60)

    # 2. Define the Camera
    # The camera requires: Position, Target (where it looks), Up vector, FOV, and Projection type
    camera = pyray.Camera3D()
    camera.position = pyray.Vector3(10.0, 10.0, 10.0)  # Camera location
    camera.target = pyray.Vector3(0.0, 0.0, 0.0)  # Camera looking at point
    camera.up = pyray.Vector3(0.0, 1.0, 0.0)  # Camera up vector (rotation towards target)
    camera.fovy = 45.0  # Camera field-of-view Y
    camera.projection = pyray.CameraProjection.CAMERA_PERSPECTIVE

    arm_angles = np.array([1, 2.0, 1.0])
    arm_lengths = np.array([1.0, 1.0, 1.0])
    arm = Arm(arm_angles, arm_lengths)

    for i in range(arm.n()):
        ic(w_i(arm,i))

    while not pyray.window_should_close():
        # --- Update ---
        # Update camera allows for orbital movement with the mouse
        pyray.update_camera(camera, pyray.CameraMode.CAMERA_ORBITAL)

        # --- Draw ---
        pyray.begin_drawing()
        pyray.clear_background(pyray.RAYWHITE)

        # Start 3D Mode
        pyray.begin_mode_3d(camera)

        # Draw a grid to visualize the ground plane (slices, spacing)
        pyray.draw_grid(10, 1.0)

        # Draw a red cube at position (0, 1, 0)
        # pyray.draw_cube(pyray.Vector3(0.0, 1.0, 0.0), 2.0, 2.0, 2.0, pyray.RED)
        # Draw wires around the cube for better visibility
        # pyray.draw_cube_wires(pyray.Vector3(0.0, 1.0, 0.0), 2.0, 2.0, 2.0, pyray.MAROON)

        # Draw a blue sphere to the side
        # pyray.draw_sphere(pyray.Vector3(4.0, 1.0, 0.0), 1.0, pyray.BLUE)

        draw_arm(arm)
        # End 3D Mode
        pyray.end_mode_3d()

        # Draw 2D UI elements (Text, FPS)
        pyray.draw_text("Drag mouse to orbit camera", 10, 40, 20, pyray.DARKGRAY)
        pyray.draw_fps(10, 10)

        pyray.end_drawing()

    # De-Initialization
    pyray.close_window()


if __name__ == "__main__":
    arm_3d()