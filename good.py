import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Callable, List, Tuple

# --- --- --- --- --- --- --- --- --- --- --- ---
# Algorithm 3.2 (from previous response)
# --- --- --- --- --- --- --- --- --- --- --- ---

def _backtracking_line_search(
        r_func: Callable[[np.ndarray], np.ndarray],
        grad_f_k: np.ndarray,
        x_k: np.ndarray,
        p_k: np.ndarray,
        c: float = 1e-4,
        tau: float = 0.5
) -> float:
    """Implements a backtracking line search."""
    r_k = r_func(x_k)
    f_k = 0.5 * np.sum(r_k**2)
    slope = np.dot(grad_f_k, p_k)
    alpha = 1.0

    while True:
        x_new = x_k + alpha * p_k
        r_new = r_func(x_new)
        f_new = 0.5 * np.sum(r_new**2)
        if f_new <= f_k + c * alpha * slope:
            return alpha
        alpha *= tau
        if alpha < 1e-9:
            return alpha

def good_deflated_gauss_newton(
        r_func: Callable[[np.ndarray], np.ndarray],
        J_func: Callable[[np.ndarray], np.ndarray],
        grad_eta_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        epsilon: float,
        tol: float = 1e-7,
        max_iter: int = 100,
        verbose: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Implements the 'good' deflated Gauss-Newton method (Algorithm 3.2)."""

    x_k = np.array(x0, dtype=float)
    history = [x_k]

    if verbose:
        print(f"--- Starting search from x0 = {x_k} ---")

    for k in range(max_iter):
        r_k = r_func(x_k)
        J_k = J_func(x_k)

        try:
            p_k, _, _, _ = np.linalg.lstsq(J_k, -r_k, rcond=None)
        except np.linalg.LinAlgError as e:
            if verbose:
                print(f"Iter {k}: Singular matrix. Stopping. Error: {e}")
            break

        step_norm = np.linalg.norm(p_k)
        if step_norm < tol:
            if verbose:
                print(f"Iter {k}: Converged (step norm {step_norm:.2e} < {tol:.2e})")
            break

        grad_eta_k = grad_eta_func(x_k)
        inner_prod = np.dot(p_k, grad_eta_k)

        deflated_step = False
        if inner_prod > epsilon:
            beta = 1.0 - inner_prod
            if np.abs(beta) > 1e-6:
                step = (1.0 / beta) * p_k
                x_k = x_k + step
                deflated_step = True
                # if verbose:
                #     print(f"Iter {k}: Deflated step (beta={beta:.2f})")

        if not deflated_step:
            grad_f_k = J_k.T @ r_k
            alpha = _backtracking_line_search(r_func, grad_f_k, x_k, p_k)
            x_k = x_k + alpha * p_k
            # if verbose:
            #     print(f"Iter {k}: Standard step (alpha={alpha:.2f})")

        history.append(x_k)

    if k == max_iter - 1 and verbose:
        print(f"Warning: Max iterations ({max_iter}) reached.")

    if verbose:
        print(f"Final solution: {x_k}\n")
    return x_k, history


# --- --- --- --- --- --- --- --- --- --- --- ---
# Test Problem Setup
# --- --- --- --- --- --- --- --- --- --- --- ---

def residual_func(x: np.ndarray) -> np.ndarray:
    """r(x,y) = [x^2 + y - 11, x + y^2 - 7]"""
    return np.array([
        x[0]**2 + x[1] - 11.0,
        x[0] + x[1]**2 - 7.0
    ])

def jacobian_func(x: np.ndarray) -> np.ndarray:
    """Jacobian of r(x,y)"""
    return np.array([
        [2.0 * x[0], 1.0],
        [1.0, 2.0 * x[1]]
    ])

def objective_func(x: np.ndarray) -> float:
    """f(x) = 0.5 * ||r(x)||^2"""
    r = residual_func(x)
    return 0.5 * np.dot(r, r)

def make_deflation_funcs(
        known_solutions: List[np.ndarray],
        sigma_sq: float = 0.5
) -> Tuple[Callable, Callable]:
    """
    Factory function to create eta and grad_eta based on a list
    of known solutions.
    """

    def eta_func(x: np.ndarray) -> float:
        """Sum of humps at known solutions."""
        eta = 0.0
        for sol in known_solutions:
            diff = x - sol
            eta += np.exp(-np.dot(diff, diff) / (2.0 * sigma_sq))
        return eta

    def grad_eta_func(x: np.ndarray) -> np.ndarray:
        """Gradient of the sum of humps."""
        grad = np.zeros_like(x)
        for sol in known_solutions:
            diff = x - sol
            hump = np.exp(-np.dot(diff, diff) / (2.0 * sigma_sq))
            grad += hump * (-1.0 / sigma_sq) * diff
        return grad

    # If no solutions are known, return functions that do nothing
    if not known_solutions:
        return (lambda x: 0.0), (lambda x: np.zeros_like(x0))

    return eta_func, grad_eta_func

# --- --- --- --- --- --- --- --- --- --- --- ---
# Main Execution
# --- --- --- --- --- --- --- --- --- --- --- ---
def deflate(func: Callable[[np.ndarray], float],
            sol: np.ndarray,
            sigma_sq: float = 0.5) -> Callable[[np.ndarray], float]:
    """Deflation function to modify the objective function."""
    def deflated_func(x: np.ndarray) -> float:
        diff = x - sol
        hump = np.exp(-np.dot(diff, diff) / (2.0 * sigma_sq))
        return func(x) * hump
    return deflated_func

if __name__ == "__main__":

    x0 = np.array([1.0, 1.0])  # Same initial guess for all runs
    known_solutions = []
    search_paths = []

    # --- Run 1: Standard Gauss-Newton (no deflation) ---
    print("=== RUN 1: Finding first solution (no deflation) ===")
    # Epsilon=inf turns off deflation
    _, grad_eta_0 = make_deflation_funcs([])
    sol_1, path_1 = good_deflated_gauss_newton(
        residual_func, jacobian_func, grad_eta_0, x0, epsilon=np.inf
    )
    known_solutions.append(sol_1)
    search_paths.append(np.array(path_1))

    # --- Run 2: Deflate sol_1 ---
    print("=== RUN 2: Finding second solution (deflating 1) ===")
    _, grad_eta_1 = make_deflation_funcs(known_solutions)
    sol_2, path_2 = good_deflated_gauss_newton(
        residual_func, jacobian_func, grad_eta_1, x0, epsilon=0.01
    )
    known_solutions.append(sol_2)
    search_paths.append(np.array(path_2))

    # --- Run 3: Deflate sol_1 and sol_2 ---
    print("=== RUN 3: Finding third solution (deflating 1, 2) ===")
    _, grad_eta_2 = make_deflation_funcs(known_solutions)
    sol_3, path_3 = good_deflated_gauss_newton(
        residual_func, jacobian_func, grad_eta_2, x0, epsilon=0.01
    )
    known_solutions.append(sol_3)
    search_paths.append(np.array(path_3))

    # --- Run 4: Deflate sol_1, sol_2, and sol_3 ---
    print("=== RUN 4: Finding fourth solution (deflating 1, 2, 3) ===")
    _, grad_eta_3 = make_deflation_funcs(known_solutions)
    sol_4, path_4 = good_deflated_gauss_newton(
        residual_func, jacobian_func, grad_eta_3, x0, epsilon=0.01
    )
    known_solutions.append(sol_4)
    search_paths.append(np.array(path_4))

    solutions = np.array(known_solutions)

    # --- --- --- --- --- --- --- --- --- --- --- ---
    # Visualization
    # --- --- --- --- --- --- --- --- --- --- --- ---
    print("--- Generating Visualization ---")

    # Create a grid for the contour plot
    x_range = np.linspace(-6, 6, 200)
    y_range = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate the objective function f(x) at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_func(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))

    # Plot the contour of f(x), using LogNorm to see the valleys
    plt.contourf(X, Y, Z, levels=np.logspace(-1, 3, 50),
                 cmap='viridis_r', norm=LogNorm())
    plt.colorbar(label=r'$f(x) = 0.5 \ ||r(x)||_2^2$')

    # Plot the search paths
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = [
        'Path 1 (Standard GN)',
        'Path 2 (Deflating sol_1)',
        'Path 3 (Deflating sol_1, 2)',
        'Path 4 (Deflating sol_1, 2, 3)'
    ]

    for i, path in enumerate(search_paths):
        plt.plot(path[:, 0], path[:, 1], 'o-',
                 color=colors[i], label=labels[i],
                 markersize=3, linewidth=2)

    # Plot the initial guess
    plt.plot(x0[0], x0[1], 'kx', markersize=12,
             markeredgewidth=3, label='Start Point')

    # Plot the solutions
    plt.plot(solutions[:, 0], solutions[:, 1], 'w*',
             markersize=15, markeredgewidth=2,
             label='Solutions')

    plt.title('Deflated Gauss-Newton Search Paths', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(loc='best')
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
