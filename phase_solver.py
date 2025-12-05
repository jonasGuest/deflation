import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from phase import generate_perfect_data, phase_residual
from good import good_deflated_gauss_newton, make_deflation_funcs, numeric_gradient


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
    n = len(x)   # Number of variables
    J = np.zeros((m, n))
    
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += h
        r_plus = r_func(x_plus)
        J[:, j] = (r_plus - r0) / h
    
    return J


def solve_acoustic_localization(
    num_solutions: int = 1,
    initial_guess: np.ndarray = None,
    visualize: bool = True
) -> list:
    """
    Find multiple acoustic source locations using deflated Gauss-Newton.
    
    Args:
        num_solutions: Number of solutions to find
        initial_guess: Starting point for optimization (2D)
        visualize: Whether to plot the results
    
    Returns:
        List of found solutions
    """
    # Get the acoustic problem setup
    mics, measured_phases, wavelength, true_source = generate_perfect_data()
    
    print(f"=== Acoustic Source Localization ===")
    print(f"Wavelength: {wavelength:.3f}m")
    print(f"Mic Spacing: {np.linalg.norm(mics[1] - mics[0]):.3f}m")
    print(f"True Source: {true_source[:2]}")
    print()
    
    # Default initial guess: center of array at 1m distance
    if initial_guess is None:
        initial_guess = np.array([0.4, 1.0])
    
    # Create residual and Jacobian functions with fixed parameters
    def r_func(x: np.ndarray) -> np.ndarray:
        return phase_residual(x, mics, measured_phases, wavelength)
    
    def J_func(x: np.ndarray) -> np.ndarray:
        return numerical_jacobian(r_func, x)
    
    # Find multiple solutions using deflation
    solutions = []
    paths = []
    
    for i in range(num_solutions):
        print(f"=== Finding Solution {i+1} ===")
        eta_func, grad_eta_func = make_deflation_funcs(solutions)
        
        sol, path = good_deflated_gauss_newton(
            r_func,
            J_func,
            grad_eta_func,
            initial_guess,
            epsilon=0.01,
            tol=1e-6,
            max_iter=100,
            verbose=True,
            use_line_search=True
        )
        
        solutions.append(sol)
        paths.append(path)
        
        # Calculate final residual
        final_residual = r_func(sol)
        final_cost = 0.5 * np.dot(final_residual, final_residual)
        print(f"Solution {i+1}: {sol}, Cost: {final_cost:.2e}")
        print()
    
    if visualize:
        visualize_solutions(mics, measured_phases, wavelength, true_source, 
                          solutions, paths, initial_guess)
    
    return solutions


def visualize_solutions(
    mics, measured_phases, wavelength, true_source,
    solutions, paths, initial_guess
):
    """Visualize the acoustic landscape with found solutions and paths."""
    
    # Create cost surface
    x_range = np.linspace(-100, 100, 500)
    y_range = np.linspace(-0.1, 200, 500)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    print("Generating cost surface for visualization...")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            guess = np.array([X[i, j], Y[i, j]])
            r = phase_residual(guess, mics, measured_phases, wavelength)
            Z[i, j] = 0.5 * np.sum(r ** 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot cost surface (log scale to see minima better)
    contour = ax.contourf(X, Y, np.log1p(Z), levels=50, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, np.log1p(Z), levels=20, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label='Log(1 + Cost)')
    
    # Plot optimization paths
    colors = ['cyan', 'magenta', 'yellow', 'lime']
    for i, path in enumerate(paths):
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'o-',
                color=colors[i % len(colors)], linewidth=2, markersize=4,
                label=f'Path {i+1}', alpha=0.8)
    
    # Plot initial guess
    ax.plot(initial_guess[0], initial_guess[1], 'kx', 
            markersize=15, markeredgewidth=3, label='Initial Guess')
    
    # Plot found solutions
    if solutions:
        sol_array = np.array(solutions)
        ax.plot(sol_array[:, 0], sol_array[:, 1], 'r*',
                markersize=20, markeredgewidth=2, label='Found Minima')
    
    # Plot true source
    ax.plot(true_source[0], true_source[1], 'g*',
            markersize=10, markeredgewidth=2, label='True Source')
    
    # Plot microphones
    ax.plot(mics[:, 0], mics[:, 1], 'k^',
            markersize=12, label='Microphones')
    
    ax.set_title('Acoustic Source Localization with Deflated Gauss-Newton\n'
                 '(Multiple Minima due to Spatial Aliasing)', fontsize=14)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-0.1, 200)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('acoustic_localization_results.png', dpi=150)
    print("Visualization saved to 'acoustic_localization_results.png'")
    plt.show()


if __name__ == "__main__":
    # Find 4 different minima starting from the same initial guess
    solutions = solve_acoustic_localization(
        num_solutions=4,
        initial_guess=np.array([1.4, 1.1]),
        visualize=True
    )
    
    print("\n=== Summary ===")
    print("All found solutions:")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: ({sol[0]:.4f}, {sol[1]:.4f})")

