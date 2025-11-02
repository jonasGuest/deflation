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

def numeric_gradient(func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-8) -> np.ndarray:
    """
    Compute the gradient of a scalar function using central finite differences.

    Args:
        func: Scalar function that takes an array and returns a float
        x: Point at which to evaluate the gradient
        h: Step size for finite differences

    Returns:
        Gradient vector at x
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (func(x_plus) - func(x_minus)) / (2.0 * h)
    return grad

def make_deflation_funcs(
        known_solutions: List[np.ndarray],
        sigma_sq: float = 0.5
) -> Tuple[Callable, Callable]:
    """
    Factory function to create eta and grad_eta based on a list
    of known solutions. Uses numeric differentiation for the gradient.
    """

    def eta_func(x: np.ndarray) -> float:
        """Sum of humps at known solutions."""
        eta = 1.0
        for sol in known_solutions:
            diff = x - sol
            eta *= (1.0+1.0/np.sqrt(np.abs(np.dot(diff, diff))))
        return eta

    def grad_eta_func(x: np.ndarray) -> np.ndarray:
        """Gradient of eta computed numerically."""
        return numeric_gradient(eta_func, x)

    return eta_func, grad_eta_func


if __name__ == "__main__":

    x0 = np.array([1.0, 1.0])  # Same initial guess for all runs

    # --- Run 1: Standard Gauss-Newton (no deflation) ---
    print("=== RUN 1: Finding first solution (no deflation) ===")
    # Epsilon=inf turns off deflation
    sols = []
    paths = []

    # Create grid for plotting
    x_range = np.linspace(-6, 6, 200)
    y_range = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x_range, y_range)

    # Create subplots: one for each deflation stage
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i in range(4):
        eta_func, grad_eta_func = make_deflation_funcs(sols)
        sol, path = good_deflated_gauss_newton(residual_func, jacobian_func, grad_eta_func, x0, epsilon=0.01)
        print("sol: ", sol)
        paths.append(path)
        sols.append(sol)

        # Compute deflated objective function on grid
        Z = np.zeros_like(X)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                x_point = np.array([X[row, col], Y[row, col]])
                r = residual_func(x_point)
                eta = eta_func(x_point)
                Z[row, col] = 0.5 * np.dot(r, r) * eta

        ax = axes[i]

        # Plot contour of deflated objective
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour, ax=ax, label='||r(x)||² * η(x)')

        # Plot all previous paths
        for j in range(i + 1):
            ax.plot(*zip(*paths[j]), 'o-',
                    label=f"Path {j}",
                    markersize=3, linewidth=2)

        # Plot the initial guess
        ax.plot(x0[0], x0[1], 'kx', markersize=12,
                markeredgewidth=3, label='Start Point')

        # Plot all found solutions
        if sols:
            ax.plot(*zip(*sols), 'r*',
                    markersize=20, markeredgewidth=2,
                    label='Solutions')

        ax.set_title(f'After Finding Solution {i}', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
