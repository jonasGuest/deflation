import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Callable, List, Tuple
import sys
import functions

def _backtracking_line_search(
        r_func: Callable[[np.ndarray], np.ndarray],
        grad_f_k: np.ndarray,
        x_k: np.ndarray,
        p_k: np.ndarray,
        c: float = 1e-4,
        tau: float = 0.5
) -> float:
    r_k = r_func(x_k)
    f_k = 0.5 * np.sum(r_k**2)

    # Compute directional derivative: slope = ∇f(x_k)^T * p_k
    slope = np.dot(grad_f_k, p_k)

    # Try more aggressive initial step size
    alpha = 2.0
    alpha_prev = 0.0
    f_prev = f_k

    for iteration in range(20):  # Max line search iterations
        # Try step x_new = x_k + alpha * p_k
        x_new = x_k + alpha * p_k
        r_new = r_func(x_new)
        f_new = 0.5 * np.sum(r_new**2)

        # Armijo condition
        if f_new <= f_k + c * alpha * slope:
            return alpha

        # Use cubic interpolation to find better alpha
        if iteration == 0:
            # First backtrack: use quadratic interpolation
            alpha_new = -0.5 * slope * alpha**2 / (f_new - f_k - slope * alpha)
        else:
            # Subsequent backtracks: use cubic interpolation
            # Fit cubic to f(0), f'(0), f(alpha_prev), f(alpha)
            a = (1.0 / (alpha - alpha_prev)) * (
                (f_new - f_k - slope * alpha) / alpha**2 -
                (f_prev - f_k - slope * alpha_prev) / alpha_prev**2
            )
            b = (1.0 / (alpha - alpha_prev)) * (
                -(f_new - f_k - slope * alpha) * alpha_prev / alpha**2 +
                (f_prev - f_k - slope * alpha_prev) * alpha / alpha_prev**2
            )

            # Solve cubic: alpha_new = (-b + sqrt(b^2 - 3*a*slope)) / (3*a)
            discriminant = b**2 - 3.0 * a * slope
            if discriminant < 0 or abs(a) < 1e-10:
                # Fall back to simple backtracking
                alpha_new = tau * alpha
            else:
                alpha_new = (-b + np.sqrt(discriminant)) / (3.0 * a)

        # Safeguard: ensure new alpha is reasonable
        alpha_new = max(0.1 * alpha, min(0.5 * alpha, alpha_new))

        # Update for next iteration
        alpha_prev = alpha
        f_prev = f_new
        alpha = alpha_new

        # Safety check: if alpha becomes too small, give up
        if alpha < 1e-9:
            return alpha

    return alpha

def good_deflated_gauss_newton(
        r_func: Callable[[np.ndarray], np.ndarray],
        J_func: Callable[[np.ndarray], np.ndarray],
        grad_eta_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        epsilon: float = 0.01,
        tol: float = 1e-7,
        max_iter: int = 100,
        verbose: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Implements the 'good' deflated Gauss-Newton method (Algorithm 3.2)."""
    x_k = np.array(x0, dtype=float)
    history = [x_k]

    if verbose:
        print(f"--- Starting search from x0 = {x_k} ---")

    k=0
    while True:
        r_k = r_func(x_k)
        J_k = J_func(x_k)

        try:
            p_k, _, _, _ = np.linalg.lstsq(J_k, -r_k, rcond=None)
        except np.linalg.LinAlgError as e:
            if verbose:
                print(f"Iter {k}: Singular matrix. Stopping. Error: {e}", file=sys.stderr)
            break

        step_norm = np.linalg.norm(p_k)
        if step_norm < tol:
            if verbose:
                print(f"Iter {k}: Converged (step norm {step_norm:.2e} < {tol:.2e})")
            break
        if k == max_iter - 1 and verbose:
            print(f"Warning: Max iterations ({max_iter}) reached.")
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

        if not deflated_step:
            grad_f_k = J_k.T @ r_k
            alpha = _backtracking_line_search(r_func, grad_f_k, x_k, p_k)
            x_k = x_k + alpha * p_k

        history.append(x_k)
        k+=1

    return x_k, history

def residual_func(x: np.ndarray) -> np.ndarray:
    """r(x,y) = [x^2 + y - 11, x + y^2 - 7]"""
    return np.array([
        x[0]**2 + x[1] - 11.0,
        x[0] + x[1]**2 - 7.0
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

def make_deflation_funcs(
        known_solutions: List[np.ndarray],
) -> Tuple[Callable, Callable]:
    def mu(x: np.ndarray) -> float:
        """Sum of humps at known solutions."""
        eta = 1.0
        for sol in known_solutions:
            diff = x - sol
            eta *= 1.0+1.0/np.abs(np.dot(diff, diff))
        return eta


    def grad_eta(x: np.ndarray) -> np.ndarray:
        """Gradient of eta computed numerically."""
        return numeric_gradient(lambda y: np.log(mu(y)), x)

    return mu, grad_eta


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
        sol, path = good_deflated_gauss_newton(residual_func, jacobian_func, grad_eta_func, x0, epsilon=0.01)
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

