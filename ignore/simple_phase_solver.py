from typing import Callable
import matplotlib.pyplot as plt
from icecream import ic
from good import make_deflation_funcs, good

import numpy as np

from utils import numerical_jacobian


def plot_residual_using_value_at_t0():
    def residual(p0: Callable[[float], float], x_i: np.ndarray, pi_t0: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def residual_func(phi: np.ndarray) -> np.ndarray:
            # phi is one dimensional but it is an array because the solver works with arrays
            assert phi.shape == (1,)
            assert(pi_t0.shape == x_i.shape)
            observed_signal_given_phi = p0(np.sin(phi[0]) * x_i / C)
            return observed_signal_given_phi - pi_t0
        return residual_func

    def residual_scalar(residual_func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray] :
        def residual_scalar_func(phi: np.ndarray) -> np.ndarray:
            assert phi.shape == (1,)
            r = residual_func(phi)
            return 1/2 * np.dot(r, r)
        return residual_scalar_func


    p0 = lambda x: np.sin(x * 2*np.pi * F)

    real_phi = np.pi / 2 * 3/4
    print("Real Phi:", real_phi)

    mic_positions = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    pi_t0 = p0(0 + np.sin(real_phi) * mic_positions / C)
    residual_func = residual(p0, mic_positions, pi_t0)
    residual_scalar_func = residual_scalar(residual_func)

    plot_phi = np.linspace(0, np.pi, 100)

    residuals_for_phi = [residual_scalar_func(np.array([phi])) for phi in plot_phi]
    print(residuals_for_phi)
    plt.plot(plot_phi, residuals_for_phi)
    plt.show()

# speed of sound in meters per second
C = 343
# Frequency in Hz
F = 8000
# Angular frequency
OMEGA = 2 * np.pi * F

def residual(x_i: np.ndarray, observed_times: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    obs_phase = (observed_times * OMEGA) % (2 * np.pi)

    def residual_func(phi: np.ndarray) -> np.ndarray:
        # phi is the estimated angle of arrival
        assert phi.shape == (1,)

        # 1. Calculate Predicted Phase (in Radians)
        # Time delay = distance / speed of sound
        pred_delay = np.sin(phi) * x_i / C
        pred_phase = pred_delay * OMEGA

        diff_cos = np.cos(pred_phase) - np.cos(obs_phase)
        diff_sin = np.sin(pred_phase) - np.sin(obs_phase)

        return np.concatenate((diff_cos, diff_sin))

    return residual_func

def residual_scalar(residual_func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def residual_scalar_func(phi: np.ndarray) -> np.ndarray:
        assert phi.shape == (1,)
        r = residual_func(phi)
        # Standard Least Squares cost: 0.5 * sum(residuals^2)
        return 0.5 * np.dot(r, r)

    return residual_scalar_func

if __name__ == "__main__":
    real_phi = np.pi / 2 * 3 / 4  # approx 1.178 radians
    print(f"Real Phi: {real_phi:.4f}")

    mic_positions = np.array([0, 0.1, 0.2, 0.25, 0.3, 0.31, 0.4, 0.5])
    pi_t0 = np.sin(real_phi) * mic_positions / C % (2*np.pi)
    residual_func = residual(mic_positions, pi_t0)
    residual_scalar_func = residual_scalar(residual_func)

    plot_phi = np.linspace(0, np.pi, 100)
    residuals_for_phi = [residual_scalar_func(np.array([phi])) for phi in plot_phi]


    phi_guess = 0.6
    sols = []
    for i in range(10):
        mu, eta_func = make_deflation_funcs(sols)
        numerical_jacobian_func = lambda x: numerical_jacobian(residual_func, x)
        found_minimum, _ = good(residual_func, numerical_jacobian_func, eta_func, np.array([phi_guess]), tol=1e-8)
        print(f"Found minimum: {found_minimum}")
        sols.append(found_minimum)
    # Find the minimum to verify it works
    min_idx = np.argmin(residuals_for_phi)
    print(f"Calculated Min Phi: {plot_phi[min_idx]:.4f}")

    plt.plot(plot_phi, residuals_for_phi)
    # plt.axvline(x=real_phi, color='r', linestyle='--', label='Actual Phi')
    for sol in sols:
        plt.axvline(x=sol[0]%2*np.pi, linestyle='--', label=f'sol {sol}')
    plt.xlabel('Angle (Radians)')
    plt.ylabel('Cost')
    plt.legend()
    plt.title('Residual Cost Function')
    plt.show()
