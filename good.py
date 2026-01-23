import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.colors import LogNorm
from typing import Callable, List, Tuple
from line_search import backtracking_line_search_wikipedia, quadratic_line_search


def find_sol(residual_func: Callable[[np.ndarray], np.ndarray], jacobian_func: Callable[[np.ndarray], np.ndarray],
             x0: np.ndarray, deflated_solutions: List[np.ndarray]) -> np.ndarray:
    _, grad_eta = make_deflation_funcs(deflated_solutions)
    new_sol, _ = good(
        r_func=residual_func,
        J_func=jacobian_func,
        grad_eta_func=grad_eta,
        x0=x0,
        verbose=False,
        max_iter=50
    )
    return new_sol

def good(
        r_func: Callable[[np.ndarray], np.ndarray],
        J_func: Callable[[np.ndarray], np.ndarray],
        grad_eta_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        epsilon: float = 0.01,
        tol: float = 1e-3,
        max_iter: int = 100,
        verbose: bool = True,
        limit_step_undeflated: float = 1.0,
        limit_step_deflated: float = 10.0,
        return_beta_instantly: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    x_k = np.array(x0, dtype=float)
    history = [x_k.copy()]
    for k in range(max_iter):
        r_k = r_func(x_k)
        J_k = J_func(x_k)
        p_k, _, _, condition_numbers = scipy.linalg.lstsq(J_k.T @ J_k, -J_k.T @ r_k)

        step_norm = np.linalg.norm(p_k)
        if step_norm < tol:
            if verbose:
                print(f"Iter {k}: Converged (step norm {step_norm:.2e} < {tol:.2e})")
            break
        if k == max_iter - 1:
            if verbose:
                print(f"Warning: Max iterations ({max_iter}) reached.")

        grad_eta_k = grad_eta_func(x_k)
        inner_prod = np.dot(p_k, grad_eta_k)
        beta = 1.0 - inner_prod

        if return_beta_instantly:
            return beta
        if inner_prod > epsilon:
            deflated_step = p_k / beta
            deflated_step_norm = np.linalg.norm(deflated_step)
            if deflated_step_norm > limit_step_deflated:
                deflated_step = np.minimum(limit_step_deflated / deflated_step_norm, 1.0) * deflated_step
            x_k = x_k + deflated_step
        else:
            p_k = np.minimum(limit_step_undeflated / step_norm, 1.0) * p_k
            pk_slope = np.dot(J_k.T @ r_k, p_k)
            scalar_r = lambda x: 0.5 * np.dot(r_func(x), r_func(x))
            alpha = backtracking_line_search_wikipedia(scalar_r, pk_slope, p_k, x_k)
            x_k = x_k + alpha * p_k

        history.append(x_k.copy())

    return x_k, history


def residual_func(x: np.ndarray) -> np.ndarray:
    """r(x,y) = [x^2 + y - 11, x + y^2 - 7]"""
    return np.array([
        x[0] ** 2 + x[1] - 11.0,
        x[0] + x[1] ** 2 - 7.0
    ])


def residual_scalar_func(x: np.ndarray) -> float:
    """Scalar objective: 0.5 * ||r(x,y)||^2"""
    r = residual_func(x)
    return 0.5 * np.dot(r, r)


def jacobian_func(x: np.ndarray) -> np.ndarray:
    """Jacobian of r(x,y)"""
    return np.array([
        [2.0 * x[0], 1.0],
        [1.0, 2.0 * x[1]]
    ])


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


def numeric_jacobian(
        r_func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        h: float = 1e-8
) -> np.ndarray:
    """
    Compute the Jacobian matrix of a vector-valued function using central finite differences.

    Args:
        r_func: Residual function that takes an array and returns a vector
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


def make_deflation_funcs(
        known_solutions: List[np.ndarray],
) -> Tuple[Callable, Callable]:
    def mu(x: np.ndarray) -> float:
        """Sum of humps at known solutions."""
        eta = 1.0
        for sol in known_solutions:
            diff = x - sol
            eta *= 1.0 + 1.0 / np.abs(np.dot(diff, diff))
        return eta

    def grad_log_eta_analytical(x: np.ndarray) -> np.ndarray:
        """
        Gradient of ln(mu(x)) computed analytically.
        Formula: Sum_k [ -2 * (x - x_k) / (dist_sq * (1 + dist_sq)) ]
        """
        grad = np.zeros_like(x)

        for sol in known_solutions:
            diff = x - sol
            dist_sq = np.dot(diff, diff)
            if dist_sq == 0:
                continue
            scalar = -2.0 / (dist_sq * (1.0 + dist_sq))
            grad += scalar * diff
        return grad

    return mu, grad_log_eta_analytical


if __name__ == "__main__":

    x0 = np.array([1.0, 1.0])  # Same initial guess for all runs

    # --- Run 1: Standard Gauss-Newton (no deflation) ---
    print("=== RUN 1: Finding first solution (no deflation) ===")
    # Epsilon=inf turns off deflation
    sols = []
    paths = []

    # Create grid for plotting
    x_range = np.linspace(-4, 4, 200)
    y_range = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x_range, y_range)

    # Create subplots: two rows (eta and deflated objective) x 4 columns (each step)
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for i in range(4):
        eta_func, grad_eta_func = make_deflation_funcs(sols)
        sol, path = good(residual_func, jacobian_func, grad_eta_func, x0, epsilon=0.01)
        print("sol: ", sol)
        paths.append(path)
        sols.append(sol)

        # Compute eta function and deflated objective function on grid
        Z_eta = np.zeros_like(X)
        Z_deflated = np.zeros_like(X)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                x_point = np.array([X[row, col], Y[row, col]])
                eta = eta_func(x_point)
                Z_eta[row, col] = eta
                Z_deflated[row, col] = residual_scalar_func(x_point)

        # Top row: eta function
        ax_eta = axes[0, i]
        contour_eta = ax_eta.contourf(X, Y, Z_eta, levels=200, cmap='plasma', alpha=0.7, norm=LogNorm())
        ax_eta.contour(X, Y, Z_eta, levels=50, colors='white', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour_eta, ax=ax_eta, label='η(x) (log scale)')

        # Plot all found solutions on eta plot
        if sols:
            ax_eta.plot(*zip(*sols), 'r',
                        markersize=20, markeredgewidth=2,
                        label='Solutions')

        ax_eta.set_title(f'η(x) After Solution {i}', fontsize=12)
        ax_eta.set_xlabel('x', fontsize=10)
        ax_eta.set_ylabel('y', fontsize=10)
        ax_eta.set_xlim(-4, 4)
        ax_eta.set_ylim(-4, 4)
        ax_eta.grid(True, linestyle='--', alpha=0.3)
        ax_eta.set_aspect('equal')
        if sols:
            ax_eta.legend(loc='upper right', fontsize=8)

        # Bottom row: deflated objective
        ax_deflated = axes[1, i]
        contour_deflated = ax_deflated.contourf(X, Y, Z_deflated, levels=50, cmap='viridis', alpha=0.7)
        ax_deflated.contour(X, Y, Z_deflated, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour_deflated, ax=ax_deflated, label='||r(x)||² * η(x)')

        # Plot all previous paths
        for j in range(i + 1):
            ax_deflated.plot(*zip(*paths[j]), 'o-',
                             label=f"Path {j}",
                             markersize=3, linewidth=2)

        # Plot the initial guess
        ax_deflated.plot(x0[0], x0[1], 'kx', markersize=12,
                         markeredgewidth=3, label='Start Point')

        # Plot all found solutions
        if sols:
            ax_deflated.plot(*zip(*sols), 'r',
                             markersize=20, markeredgewidth=2,
                             label='Solutions')

        ax_deflated.set_title(f'Deflated Objective After Solution {i}', fontsize=12)
        ax_deflated.set_xlabel('x', fontsize=10)
        ax_deflated.set_ylabel('y', fontsize=10)
        ax_deflated.legend(loc='best', fontsize=8)
        ax_deflated.set_xlim(-4, 4)
        ax_deflated.set_ylim(-4, 4)
        ax_deflated.grid(True, linestyle='--', alpha=0.3)
        ax_deflated.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    print(f"All found solutions: {sols}")
