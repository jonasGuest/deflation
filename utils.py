from typing import Callable

import numpy as np


def numerical_jacobian_func(
        r_func: Callable[[np.ndarray], np.ndarray],
        h: float = 1e-8
) -> Callable[[np.ndarray], np.ndarray]:
    def l( x: np.ndarray):
        return numerical_jacobian(r_func, x, h)
    return l

# two sided numerical jacobian
def numerical_jacobian(
        r_func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        h: float = 1e-8
) -> np.ndarray:
    """
    Compute the Jacobian matrix using finite differences.

    Args:
        r_func: Residual function that returns a vector
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences

    Returns:
        Jacobian matrix J where J[i,j] = ∂r_i/∂x_j
    """
    r0 = r_func(x)
    m = len(r0)  # Number of residuals
    n = len(x)  # Number of variables
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        r_plus = r_func(x_plus)
        r_minus = r_func(x_minus)
        J[:, j] = (r_plus - r_minus) / (2.0 * h)

    return J
