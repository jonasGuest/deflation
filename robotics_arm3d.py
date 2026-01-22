from typing import List, Callable, Tuple
import numpy as np
import pyray
import time
from good import find_sol

class Arm:
    def __init__(self, angles: np.ndarray, rotation_axis: List[np.ndarray], r_i: List[np.ndarray]):
        assert angles.shape[0] == len(r_i)
        assert len(rotation_axis) == angles.shape[0]
        self.angles = angles
        self.rotation_axis = rotation_axis
        self.r_i = r_i

    def with_angles(self, angles: np.ndarray):
        return Arm(angles, self.rotation_axis, self.r_i)

    def n(self):
        return self.angles.shape[0]


z_axis = np.array([0.0, 0.0, 1.0])
y_axis = np.array([0.0, 1.0, 0.0])
x_axis = np.array([1.0, 0.0, 0.0])


def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    return np.eye(3, 3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def w_i(arm: Arm, i: int) -> np.ndarray:
    w = arm.rotation_axis[i]
    for j in range(i - 1, -1, -1):
        w = rot(arm.rotation_axis[j], arm.angles[j]) @ w
    return w


def jacobian(arm: Arm) -> np.ndarray:
    J = np.zeros(shape=(3, arm.n()))
    end_effector_pos = s_i(arm, arm.n())

    for i in range(arm.n()):
        J[:, i] = np.cross(w_i(arm, i), (end_effector_pos - s_i(arm, i)))
    return J


def s_i(arm: Arm, i: int) -> np.ndarray:
    tip = np.array([0., 0., 0.])
    for j in range(i - 1, -1, -1):
        tip = rot(arm.rotation_axis[j], arm.angles[j]) @ (tip + arm.r_i[j])

    return tip


def lerp(current: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    return current * (1 - t) + target * t


# pyray ----------------------
def np_to_pyray(vec: np.ndarray) -> pyray.Vector3:
    return pyray.Vector3(vec[0], vec[1], vec[2])


def draw_arm(arm: Arm, is_primary: bool = True):
    if is_primary:
        joint_color = pyray.Color(0, 0, 0, 255)
        segment_color = pyray.Color(20, 20, 20, 200)
    else:
        joint_color = pyray.Color(255, 255, 0, 100)
        segment_color = pyray.Color(255, 255, 0, 100)
    segment_radius = 0.08
    joint_radius = 0.15

    for i in range(arm.n()):
        current_pos = np_to_pyray(s_i(arm, i))
        pyray.draw_sphere(current_pos, joint_radius, joint_color)

        # Draw link to next joint
        next_pos = np_to_pyray(s_i(arm, i + 1))
        pyray.draw_cylinder_ex(current_pos, next_pos, segment_radius, segment_radius, 16, segment_color)

    # Draw end effector tip
    if is_primary:
        pyray.draw_sphere(np_to_pyray(s_i(arm, arm.n())), joint_radius * 0.8, pyray.BLACK)


def select_primary_solution(solutions: List[np.ndarray], current_arm: Arm, target_pos: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    assert len(solutions) > 0, "No solutions provided"
    assert len(solutions[0]) == len(current_arm.angles)
    assert len(target_pos) == 3

    def cost(solution: np.ndarray) -> float:
        angle_diff = np.linalg.norm(current_arm.angles - solution) ** 2
        temp_arm = current_arm.with_angles(solution)
        pos_diff = np.linalg.norm(s_i(temp_arm, temp_arm.n()) - target_pos) ** 2
        return float(angle_diff + 10 * pos_diff)
    best_solution = solutions[0]
    min_cost = cost(best_solution)
    for solution in solutions[1:]:
        c = cost(solution)
        if c < min_cost:
            min_cost = c
            best_solution = solution
    return best_solution, [sol for sol in solutions if not np.array_equal(sol, best_solution)]

class MyCamera:
    def __init__(self):
        self.orbit_center = np.array([0.0, 2.0, 0.0])
        self.radius = 13.0
        self.yaw = np.pi / 4.0
        self.pitch = np.pi / 9.0
        self.rot_speed = 0.02

        self.camera = pyray.Camera3D()
        self.camera.target = np_to_pyray(self.orbit_center)
        self.camera.up = pyray.Vector3(0.0, 1.0, 0.0)
        self.camera.fovy = 45.0
        self.camera.projection = pyray.CameraProjection.CAMERA_PERSPECTIVE

    def a_pressed(self):
        self.yaw -= self.rot_speed

    def d_pressed(self):
        self.yaw += self.rot_speed

    def w_pressed(self):
        self.pitch += self.rot_speed

    def s_pressed(self):
        self.pitch -= self.rot_speed

    def update(self):
        self.pitch = np.clip(self.pitch, 0.1, np.pi / 2 - 0.1)
        new_cam_x = self.orbit_center[0] + self.radius * np.cos(self.pitch) * np.cos(self.yaw)
        new_cam_z = self.orbit_center[2] + self.radius * np.cos(self.pitch) * np.sin(self.yaw)
        new_cam_y = self.orbit_center[1] + self.radius * np.sin(self.radius)

        self.camera.position = pyray.Vector3(new_cam_x, new_cam_y, new_cam_z)


def arm_3d():
    pyray.init_window(1200, 800, "IK with Levenberg-Marquardt (Vertical Interaction Plane)")
    pyray.set_target_fps(60)

    camera = MyCamera()
    initial_angles = np.array([-0.5, 0.5, 0.5, 0.5])
    arm_rotation_axis = [y_axis, z_axis, z_axis, z_axis]
    r_i = [
        np.array([0.0, 0.5, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    ]

    arm = Arm(initial_angles, arm_rotation_axis, r_i)
    target_pos = np.array([-3.0, 1.0, -1.0])

    def jacobian_func(x):
        return jacobian(arm.with_angles(x))

    secondary_solutions = find_multiple_solutions(arm, jacobian_func, target_pos)

    start_angles = initial_angles.copy()
    if len(secondary_solutions) > 0:
        target_angles, secondary_solutions = select_primary_solution(secondary_solutions, arm, target_pos)
    else:
        target_angles = initial_angles.copy()

    animation_duration = 0.4
    animation_start = time.time()
    last_time_target_selected = time.time()

    while not pyray.window_should_close():
        if pyray.is_key_down(pyray.KeyboardKey.KEY_A):
            camera.a_pressed()
        if pyray.is_key_down(pyray.KeyboardKey.KEY_D):
            camera.d_pressed()
        if pyray.is_key_down(pyray.KeyboardKey.KEY_W):
            camera.w_pressed()
        if pyray.is_key_down(pyray.KeyboardKey.KEY_S):
            camera.s_pressed()

        camera.update()

        pyray.begin_drawing()
        pyray.clear_background(pyray.RAYWHITE)
        pyray.begin_mode_3d(camera.camera)

        pyray.draw_grid(20, 1.0)
        pyray.draw_sphere(np_to_pyray(target_pos), 0.18, pyray.GREEN)

        pyray.draw_line_3d(pyray.Vector3(0, 0, 0), pyray.Vector3(2, 0, 0), pyray.RED)
        pyray.draw_line_3d(pyray.Vector3(0, 0, 0), pyray.Vector3(0, 2, 0), pyray.GREEN)
        pyray.draw_line_3d(pyray.Vector3(0, 0, 0), pyray.Vector3(0, 0, 2), pyray.BLUE)

        # --- Interaction Logic ---
        now = time.time()
        if pyray.is_mouse_button_down(pyray.MouseButton.MOUSE_BUTTON_LEFT) and now - last_time_target_selected > 0.2:
            last_time_target_selected = now
            ray = pyray.get_screen_to_world_ray(pyray.get_mouse_position(), camera.camera)

            plane_normal = np.array([camera.camera.position.x, 0.0, camera.camera.position.z])
            norm_len = np.linalg.norm(plane_normal)

            if norm_len > 1e-6:
                plane_normal /= norm_len

                ro = np.array([ray.position.x, ray.position.y, ray.position.z])
                rd = np.array([ray.direction.x, ray.direction.y, ray.direction.z])

                denom = np.dot(rd, plane_normal)
                if abs(denom) > 1e-6:
                    t = -np.dot(ro, plane_normal) / denom

                    if t > 0:
                        hit = ro + t * rd
                        target_pos = hit

                        # Snap start_angles to current visual position to prevent jumping
                        elapsed = now - animation_start
                        animation_percent = min(elapsed / animation_duration, 1.0)
                        start_angles = lerp(start_angles, target_angles, animation_percent)

                        # Update arm internal angles for the solver (start search from current visual)
                        arm.angles = start_angles

                        # Solve IK
                        new_solutions = find_multiple_solutions(arm, jacobian_func, target_pos)
                        if len(new_solutions) > 0:
                            target_angles, secondary_solutions = select_primary_solution(new_solutions, arm,
                                                                                         target_pos)
                            animation_start = now

        # --- Animation Logic ---
        current_time = time.time()
        elapsed = current_time - animation_start
        animation_percent = min(elapsed / animation_duration, 1.0)

        # Calculate visual angles for this frame
        draw_angles = lerp(start_angles, target_angles, animation_percent)

        # Draw Secondary (Ghost) Solutions
        for solution in secondary_solutions:
            draw_arm(arm.with_angles(solution), False)

        # Draw Primary Arm
        draw_arm(arm.with_angles(draw_angles), True)

        pyray.end_mode_3d()

        pyray.draw_text("Target Control: Click to move on vertical facing plane", 10, 10, 20, pyray.DARKGRAY)
        pyray.draw_fps(10, 70)

        pyray.end_drawing()

    pyray.close_window()


def find_multiple_solutions(arm: Arm, jacobian_func: Callable[[np.ndarray], np.ndarray], target_pos: np.ndarray):
    def residuals_func_dynamic(x):
        return s_i(arm.with_angles(x), arm.n()) - target_pos

    solutions = []

    for i in range(3):
        new_angles = find_sol(residuals_func_dynamic, jacobian_func, arm.angles, solutions)
        if new_angles is not None:
            solutions.append(new_angles)
    return solutions


if __name__ == "__main__":
    arm_3d()
