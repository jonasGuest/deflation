from typing import Callable

import numpy as np
import pyray
from utils import numerical_jacobian, numerical_jacobian_func
import time


def robotic_arm(origin: np.ndarray, lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """ Calculate the (x, y) position of the end effector of a 2D robotic arm. """
    tip = origin.copy()
    for length, angle in zip(lengths, angles):
        tip[0] += length * np.cos(angle)
        tip[1] += length * np.sin(angle)

    return tip

def residuals(origin: np.ndarray, lengths: np.ndarray, target: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def l(angles: np.ndarray):
        return robotic_arm(origin, lengths, angles) - target
    return l

def find_minimum_of_residuals(origin: np.ndarray, lengths: np.ndarray, initial_angles: np.ndarray, target: np.ndarray) -> np.ndarray:
    from good import good, make_deflation_funcs
    _, grad_eta = make_deflation_funcs([])
    residuals_func = residuals(origin, lengths, target)
    angles, _ = good(residuals_func,
                  numerical_jacobian_func(residuals_func),
                  grad_eta,
                  initial_angles
                  )
    return angles

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def lerp_angles(current: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    return np.array([
        lerp(current[0], target[0], t),
        lerp(current[1], target[1], t)
    ])


def render_double_arm(origin: np.ndarray, lengths: np.ndarray, angles: np.ndarray):
    joints = [origin]
    for i in range(np.shape(angles)[0]):
        joints.append(joints[-1] + np.array([
            lengths[i] * np.cos(angles[i]),
            lengths[i] * np.sin(angles[i])
        ]))

    for i in range(len(joints) - 1):
        start = joints[i]
        end = joints[i + 1]
        pyray.draw_line_ex(start.tolist(), end.tolist(), 4.5, pyray.DARKGRAY)
        pyray.draw_circle(int(start[0]), int(start[1]), 10, pyray.BLACK)

if __name__ == "__main__":
    # Example usage
    lengths = np.array([50.0, 100.0])

    pyray.init_window(800, 450, "Demo - Python Raylib")
    pyray.set_target_fps(60)

    initial_anchor = np.array([400.0, 225.0])

    target = np.array([500.0, 300.0])

    angles = np.array([0.0, 0.0])
    origin = np.array([400, 200])


    # Initial Solve
    target_angles = find_minimum_of_residuals(origin, lengths, angles, target)
    print(f"Optimized angles: {target_angles}")

    animation_duration = 0.4  # seconds (converted from 400.0ms)
    animation_start = time.time()

    while not pyray.window_should_close():

        # Input Handling
        if pyray.is_mouse_button_pressed(pyray.MouseButton.MOUSE_BUTTON_LEFT):
            mouse_pos = pyray.get_mouse_position()
            target = np.array([float(mouse_pos.x), float(mouse_pos.y)])

            # Snap last_angles to current visual state before starting new animation
            current_time = time.time()
            elapsed = current_time - animation_start
            animation_percent = min(elapsed / animation_duration, 1.0)
            angles = lerp_angles(angles, target_angles, animation_percent)

            target_angles = find_minimum_of_residuals(origin, lengths, angles, target)

            print(f"New target: ({target[0]}, {target[1]})")
            print(f"Optimized angles: {target_angles}")

            animation_start = current_time

        # Animation Logic
        current_time = time.time()
        elapsed = current_time - animation_start
        animation_percent = min(elapsed / animation_duration, 1.0)
        draw_angles = lerp_angles(angles, target_angles, animation_percent)

        # Rendering
        pyray.begin_drawing()
        pyray.clear_background(pyray.RAYWHITE)

        # Draw Target
        pyray.draw_circle_v(target.tolist(), 20, pyray.GREEN)

        # Draw Arm
        render_double_arm(origin, lengths, draw_angles)

        # Draw UI
        pyray.draw_grid(20, 1.0)
        pyray.draw_text("Click to move target", 190, 200, 20, pyray.VIOLET)
        pyray.draw_fps(20, 20)

        pyray.end_drawing()

    pyray.close_window()


if __name__ == "__main__":
    main()