import numpy as np
from typing import Callable
import sys
from scipy.optimize import minimize_scalar

def backtracking_line_search_wikipedia_old(f: Callable[[np.ndarray], float],
                                       slope_p: float,
                                       p: np.ndarray,
                                       x: np.ndarray,
                                       alpha_0: float = 1.0,
                                       tau=0.5, c=1e-4,
                                       max_iter: int = 1000) -> float:
    t = c*slope_p
    alpha = alpha_0
    fx = f(x)
    for j in range(max_iter):
        if (fx + alpha * t) > f(x+alpha * p):
            return alpha
        alpha = tau * alpha
    print("No convergence backtracking line search", file=sys.stderr)
    return alpha

def backtracking_line_search_wikipedia(f: Callable[[np.ndarray], float],
                                       slope_p: float,
                                       p: np.ndarray,
                                       x: np.ndarray,
                                       alpha_0: float = 1.0,
                                       tau=0.5, c=1e-4,
                                       max_iter: int = 100) -> float:
    t = c*slope_p
    alpha = alpha_0
    fx = f(x)
    for j in range(max_iter):
        if (fx + alpha * t) > f(x+alpha * p):
            return alpha
        alpha = tau * alpha
    print("No convergence backtracking line search", file=sys.stderr)
    return alpha

def quadratic_line_search(f: Callable[[np.ndarray], float],
                          slope_p: float,
                          p: np.ndarray,
                          x: np.ndarray,
                          alpha_0: float = 1.0,
                          max_iter: int = 100) -> float:
    res = minimize_scalar(lambda y: f(y * p + x), method='brent')
    if not res.success:
        print("No convergence in quadratic line search", file=sys.stderr)
    return res.x



if __name__ == "__main__":
    def f1(x: np.ndarray) -> float:
        # 0.6 x4 + 3.4 x3 + 4.1 x2−0.3 x + 1
        assert (x.shape == (1,))
        return 0.6 * x[0] ** 4 + 3.4 * x[0] ** 3 + 4.1 * x[0] ** 2 - 0.3 * x[0] + 1
    def df1(x: np.ndarray) -> float:
        # Derivative: 2.4 x3 + 10.2 x2 + 8.2 x - 0.3
        assert (x.shape == (1,))
        return 2.4 * x[0] ** 3 + 10.2 * x[0] ** 2 + 8.2 * x[0] - 0.3

    start1 = np.array([-1.5])
    p1 = np.array([-1.0])
    p1_slope = df1(start1) * p1[0]

    start2 = np.array([-0.2])
    p2 = np.array([1.0])
    p2_slope = df1(start2) * p2[0]

    alpha1 = backtracking_line_search_wikipedia(f1, p1_slope, p1, start1)
    alpha2 = backtracking_line_search_wikipedia(f1, p2_slope, p2, start2)
    print(f"Optimal step size 1: {alpha1}, to reach {start1 + alpha1 * p1}")
    print(f"Optimal step size 2: {alpha2}, to reach {start2 + alpha2 * p2}")

    # Example usage
    def example_func(x):
        return x[0]**2 + x[1]**2

    x0 = np.array([0.2, 0.2])
    p = np.array([-1.0, -1.0])
    slope_p = -2 * np.dot(x0, p)  # Gradient dot direction

    alpha = backtracking_line_search_wikipedia(example_func, slope_p, p, x0)
    print(f"Optimal step size: {alpha}")