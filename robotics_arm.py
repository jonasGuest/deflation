from typing import List, Callable

import numpy as np
import numpy.linalg
import pyray
import scipy.optimize

from utils import numerical_jacobian
from good import good, make_deflation_funcs
import time

PYRAY_SECONDARY_COLORS=[
    pyray.GREEN, pyray.YELLOW, pyray.PURPLE
]
SHOW_SECONDARY_SOLUTIONS = True

def robotic_arm(origin: np.ndarray, lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """ Calculate the (x, y) position of the end effector of a 2D robotic arm. """
    tip = origin.copy()
    for length, angle in zip(lengths, angles):
        tip[0] += length * np.cos(angle)
        tip[1] += length * np.sin(angle)

    return tip


def residuals(origin: np.ndarray, lengths: np.ndarray, target: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def residuals(angles: np.ndarray) -> np.ndarray:
        end_effector_pos = robotic_arm(origin, lengths, angles)
        residual_value = np.zeros(2)
        # prefer small angles
        residual_value[0:2] = end_effector_pos - target
        # This is very sensitive when trying to encode additional things into the residuals the accuracy suffers hugely
        # residual_value[2] = end
        # residual_value[3] = 0
        return residual_value

    return residuals


def simple_gauss_newton(residuals: Callable[[np.ndarray], np.ndarray], jacobian: Callable[[np.ndarray], np.ndarray],
                        guess: np.ndarray, tol=1e-4, max_iter=100) -> np.ndarray:
    x = guess.copy()
    for i in range(max_iter):
        j = jacobian(x)
        r = residuals(x)
        jTj = j.T @ j
        try:
            p = scipy.linalg.solve(jTj, -j.T @ r)
        except numpy.linalg.LinAlgError as e:
            print(e)
            p = np.zeros_like(x)
            p[0] = 0.1

        x += p
        if np.linalg.norm(p) < tol:
            break
        if i == max_iter - 1:
            print("reached maximum iteration")
    return x


def find_minimum_of_residuals(origin: np.ndarray, lengths: np.ndarray, initial_angles: np.ndarray,
                              target: np.ndarray) -> List[np.ndarray]:
    sols = []
    residuals_func = residuals(origin, lengths, target)
    jacobian_func = lambda x: numerical_jacobian(residuals_func, x)
    for i in range(3):
        _, grad_eta = make_deflation_funcs(sols)
        angles, path = good(residuals_func, jacobian_func, grad_eta, initial_angles)
        sols.append(angles)
    return [sol % (2 * np.pi) for sol in sols]



def choose_minimum_solution(origin: np.ndarray, lengths: np.ndarray, target: np.ndarray,
                            solutions: List[np.ndarray]) -> np.ndarray:
    best_solution = solutions[0]
    residual_func = residuals(origin, lengths, target)
    for sol in solutions[1:]:
        if np.dot(residual_func(sol), residual_func(sol)) < np.dot(residual_func(best_solution),
                                                                   residual_func(best_solution)):
            best_solution = sol
    return best_solution


def lerp_angles(current: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    return current * (1 - t) + target * t


def render_arm(origin: np.ndarray, lengths: np.ndarray, angles: np.ndarray, color: pyray.Color=pyray.BLACK, opacity:float = 1):
    joints = [origin]
    for i in range(np.shape(angles)[0]):
        joints.append(joints[-1] + np.array([
            lengths[i] * np.cos(angles[i]),
            lengths[i] * np.sin(angles[i])
        ]))

    for i in range(len(joints) - 1):
        start = joints[i]
        end = joints[i + 1]
        pyray.draw_line_ex(start.tolist(), end.tolist(), 4.5, pyray.fade(color, opacity))
        pyray.draw_circle(int(start[0]), int(start[1]), 10, pyray.fade(color, opacity))


if __name__ == "__main__":
    # Example usage
    lengths = np.array([50.0, 100.0, 75.0, 30, 20])

    pyray.init_window(800, 450, "Demo - Python Raylib")
    pyray.set_target_fps(60)

    target = np.array([500.0, 300.0])

    angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    origin = np.array([400.0, 200.0])

    # Initial Solve
    multiple_solutions = find_minimum_of_residuals(origin, lengths, angles, target)
    print(multiple_solutions)
    target_angles = choose_minimum_solution(origin, lengths, target, multiple_solutions)

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

            multiple_solutions = find_minimum_of_residuals(origin, lengths, angles, target)
            target_angles = choose_minimum_solution(origin, lengths, target, multiple_solutions)

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
        pyray.draw_circle_v(target.tolist(), 3, pyray.GREEN)

        # Draw Arm
        render_arm(origin, lengths, draw_angles)

        # Draw UI
        pyray.draw_grid(20, 1.0)
        pyray.draw_text("Click to move target", 190, 200, 20, pyray.VIOLET)
        pyray.draw_fps(20, 20)

        for i,solution in enumerate(multiple_solutions):
            tip = robotic_arm(origin, lengths, solution)
            if SHOW_SECONDARY_SOLUTIONS:
                render_arm(origin, lengths, solution, color=PYRAY_SECONDARY_COLORS[i], opacity=0.2)

        pyray.end_drawing()

    pyray.close_window()

