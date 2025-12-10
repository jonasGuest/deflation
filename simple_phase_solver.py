from typing import Callable
import matplotlib.pyplot as plt
from icecream import ic

import numpy as np


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

# speed of sound in meters per secdond
C = 343
# 5000 Hz
F = 8000

def residual(x_i: np.ndarray, pi_t0: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    # pi_t0 are the observed phases
    def residual_func(phi: np.ndarray) -> np.ndarray:
        # phi is one dimensional but it is an array because the solver works with arrays
        assert phi.shape == (1,)
        assert (pi_t0.shape == x_i.shape)
        phase_given_phi = (np.sin(phi) * x_i / C + pi_t0[0]) % (2*np.pi / F)
        return np.concatenate((np.sin(phase_given_phi) - np.sin(pi_t0), np.cos(phase_given_phi) - np.cos(pi_t0)))

    return residual_func


def residual_scalar(residual_func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def residual_scalar_func(phi: np.ndarray) -> np.ndarray:
        assert phi.shape == (1,)
        r = residual_func(phi)
        return 1 / 2 * np.dot(r, r)

    return residual_scalar_func


if __name__ == "__main__":

    real_phi = np.pi / 2 * 3 / 4
    print("Real Phi:", real_phi)

    mic_positions = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    pi_t0 = np.sin(real_phi) * mic_positions / C % (2*np.pi / F)
    residual_func = residual(mic_positions, pi_t0)
    residual_scalar_func = residual_scalar(residual_func)

    plot_phi = np.linspace(0, np.pi, 100)

    residuals_for_phi = [residual_scalar_func(np.array([phi])) for phi in plot_phi]
    print(residuals_for_phi)
    plt.plot(plot_phi, residuals_for_phi)
    plt.show()

