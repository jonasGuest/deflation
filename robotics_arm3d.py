from typing import List, Callable, Tuple

import numpy as np
import numpy.linalg
import pyray
import time
from good import make_deflation_funcs, good, find_sol

class Arm:
    def __init__(self, angles: np.ndarray, lengths: np.ndarray, rotation_axis: List[np.ndarray]):
        assert angles.shape == lengths.shape
        assert len(rotation_axis) == angles.shape[0]
        self.angles = angles
        self.lengths = lengths
        self.rotation_axis = rotation_axis
    def with_angles(self, angles: np.ndarray):
        return Arm(angles, self.lengths, self.rotation_axis)
    def n(self):
        return self.angles.shape[0]


# TODO make axis variable. hardcode axis of rotation to z_axis
z_axis = np.array([0.0, 0.0, 1.0])

def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
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


# pyray ----------------------
def np_to_pyray(vec: np.ndarray) -> pyray.Vector3:
    return pyray.Vector3(vec[0], vec[1], vec[2])

def draw_arm(arm: Arm, is_primary: bool = True):
    # if is_primary:
    #     joint_color = pyray.MAROON
    #     segment_color = pyray.DARKGRAY
    # else:
    if is_primary:
        joint_color = pyray.Color(255, 255, 0, 100)
        segment_color = pyray.Color(255, 255, 0, 100)
    else :
        joint_color = pyray.Color(0, 0, 255, 200)
        segment_color = pyray.Color(0, 0, 200, 100)
    segment_radius = 0.1
    joint_radius = 0.2

    for i in range(arm.n()):
        current_pos = np_to_pyray(s_i(arm, i))
        pyray.draw_sphere(current_pos, joint_radius, joint_color)

        # Draw link to next joint
        next_pos = np_to_pyray(s_i(arm, i + 1))
        pyray.draw_cylinder_ex(current_pos, next_pos, segment_radius, segment_radius, 16, segment_color)

    # Draw end effector tip
    if is_primary:
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
    arm_rotation_axis = [z_axis, z_axis, z_axis]

    arm = Arm(arm_angles, arm_lengths, arm_rotation_axis)

    target_pos = np.array([-3.0, 1.0, 0.0])

    def residuals_func(x):
        return s_i(arm.with_angles(x), arm.n()) - target_pos

    def jacobian_func(x):
        return jacobian(arm.with_angles(x))

    mu, grad_eta = make_deflation_funcs([])
    new_angles, path = good(
        r_func=residuals_func,
        J_func=jacobian_func,
        grad_eta_func=grad_eta,
        x0=arm.angles
    )
    arm.angles = new_angles

    solutions = find_multiple_solutions(arm, jacobian_func, target_pos)

    while not pyray.window_should_close():
        # --- Update ---
        # pyray.update_camera(camera, pyray.CameraMode.CAMERA_ORBITAL)

        # --- 3. Draw ---
        pyray.begin_drawing()
        pyray.clear_background(pyray.RAYWHITE)
        pyray.begin_mode_3d(camera)

        pyray.draw_grid(20, 1.0)
        pyray.draw_sphere(np_to_pyray(target_pos), 0.3, pyray.GREEN)

        # --- Mouse Interaction ---
        if pyray.is_mouse_button_down(pyray.MouseButton.MOUSE_BUTTON_LEFT):
            ray = pyray.get_screen_to_world_ray(pyray.get_mouse_position(), camera)

            if abs(ray.direction.z) > 1e-6:
                t = -ray.position.z / ray.direction.z

                # Check if the intersection is in front of the camera
                if t > 0:
                    hit_x = ray.position.x + ray.direction.x * t
                    hit_y = ray.position.y + ray.direction.y * t

                    target_pos = np.array([hit_x, hit_y, 0.0])

                    # 3. Re-run IK
                    # We redefine the residual function to use the NEW target_pos
                    solutions = find_multiple_solutions(arm, jacobian_func, target_pos)
                    # print("solutions: ", solutions)
                    arm.angles = solutions[0]

        for i,solution in enumerate(solutions):
            print(i, solution)
            other_arm = arm.with_angles(solution)
            draw_arm(other_arm, i==0)

        print()

        pyray.end_mode_3d()

        # UI
        pyray.draw_text("Target Control: W/A/S/D (Planar), Q/E (Height)", 10, 10, 20, pyray.DARKGRAY)
        pyray.draw_text(f"Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]", 10, 40, 20, pyray.BLACK)
        pyray.draw_fps(10, 70)

        pyray.end_drawing()

    pyray.close_window()


def find_multiple_solutions(arm: Arm, jacobian_func: Callable[[np.ndarray], np.ndarray], target_pos: np.ndarray):
    def residuals_func_dynamic(x):
        return s_i(arm.with_angles(x), arm.n()) - target_pos

    solutions = []
    for i in range(3):
        new_angles = find_sol(residuals_func_dynamic, jacobian_func, arm.angles, solutions)
        solutions.append(new_angles)
    return solutions


if __name__ == "__main__":
    arm_3d()
