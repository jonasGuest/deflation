import good
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def scalar_to_residual(f):
    """Convert scalar function to residual format."""

    def residual(x):
        return np.array([f(x)])

    return residual


def scalar_to_jacobian(f):
    """Convert scalar function to Jacobian format."""

    def jacobian(x):
        grad = good.numeric_gradient(f, x)
        return grad.reshape(1, -1)  # 1×n matrix

    return jacobian


def test_acoustic_phase_residual_func():
    """Test deflation on the acoustic_phase_residual function."""
    from functions import test_problem_with_many_local_minima

    # Convert residual function to scalar cost function
    def acoustic_cost(x):
        r = test_problem_with_many_local_minima(x)
        return 0.5 * np.dot(r, r)

    # Initial guess
    x0 = np.array([0.0, 0.0])

    # Storage for solutions and paths
    solutions = []
    paths = []

    # Number of deflation steps
    n_deflations = 4

    # Create grid for plotting
    x_range = np.linspace(-2000, 2000, 300)
    y_range = np.linspace(-2000, 2000, 300)
    X, Y = np.meshgrid(x_range, y_range)

    # Compute base acoustic cost on grid
    Z_base = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_base[i, j] = acoustic_cost(np.array([X[i, j], Y[i, j]]))

    # Create subplots: 3 rows x n_deflations columns
    fig, axes = plt.subplots(3, n_deflations, figsize=(7 * n_deflations, 18))

    if n_deflations == 1:
        axes = axes.reshape(-1, 1)

    print("=== Testing Acoustic Phase Residual with Deflation ===\n")

    for step in range(n_deflations):
        print(f"--- Deflation Step {step + 1} ---")

        # Create deflation functions
        eta_func, grad_eta_func = good.make_deflation_funcs(solutions)

        # Run optimization - use residual format
        def acoustic_residual(x):
            return test_problem_with_many_local_minima(x)

        def acoustic_jacobian(x):
            return good.numeric_jacobian(acoustic_residual, x)

        sol, path = good.good(
            acoustic_residual,
            acoustic_jacobian,
            grad_eta_func,
            x0,
            epsilon=0.01,
            max_iter=100,
            verbose=True,
            use_line_search=True
        )

        print(f"Solution {step + 1}: {sol}")
        print(f"Cost value: {acoustic_cost(sol):.6e}\n")

        solutions.append(sol)
        paths.append(path)

        # Compute eta and deflated function on grid
        Z_eta = np.zeros_like(X)
        Z_deflated = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_point = np.array([X[i, j], Y[i, j]])
                eta = eta_func(x_point)
                Z_eta[i, j] = eta
                Z_deflated[i, j] = acoustic_cost(x_point) * eta

        # Row 0: Base function with all paths
        ax_base = axes[0, step]
        # Use log scale for better visualization
        Z_base_log = np.log1p(Z_base)
        contour_base = ax_base.contourf(X, Y, Z_base_log, levels=50, cmap='viridis', alpha=0.7)
        ax_base.contour(X, Y, Z_base_log, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour_base, ax=ax_base, label='log(1 + Cost)')

        # Plot all paths
        for i in range(step + 1):
            path_array = np.array(paths[i])
            ax_base.plot(path_array[:, 0], path_array[:, 1], 'o-',
                         label=f"Path {i + 1}", markersize=3, linewidth=2, alpha=0.8)

        # Plot initial guess and solutions
        ax_base.plot(x0[0], x0[1], 'kx', markersize=15,
                     markeredgewidth=3, label='Start', zorder=10)

        if solutions:
            sols_array = np.array(solutions)
            ax_base.scatter(sols_array[:, 0], sols_array[:, 1],
                            c='red', s=200, marker='*', edgecolors='white',
                            linewidths=2, label='Solutions', zorder=10)

        ax_base.set_title(f'Acoustic Cost + Paths (Step {step + 1})',
                          fontsize=12, fontweight='bold')
        ax_base.set_xlabel('x', fontsize=10)
        ax_base.set_ylabel('y', fontsize=10)
        ax_base.set_xlim(-2000, 2000)
        ax_base.set_ylim(-2000, 2000)
        ax_base.grid(True, linestyle='--', alpha=0.3)
        ax_base.set_aspect('equal')
        ax_base.legend(loc='upper right', fontsize=8)

        # Row 1: Eta function
        ax_eta = axes[1, step]
        Z_eta_clipped = np.minimum(Z_eta, 100)
        contour_eta = ax_eta.contourf(X, Y, Z_eta_clipped, levels=200,
                                      cmap='plasma', alpha=0.7, norm=LogNorm())
        ax_eta.contour(X, Y, Z_eta_clipped, levels=50, colors='white',
                       alpha=0.3, linewidths=0.5)
        plt.colorbar(contour_eta, ax=ax_eta, label='η(x) (log scale, clipped)')

        if solutions:
            sols_array = np.array(solutions)
            ax_eta.scatter(sols_array[:, 0], sols_array[:, 1],
                           c='red', s=200, marker='*', edgecolors='white',
                           linewidths=2, label='Solutions', zorder=10)

        path_array = np.array(paths[step])
        ax_eta.plot(path_array[:, 0], path_array[:, 1], 'wo-',
                    markersize=3, linewidth=2, alpha=0.6, label=f'Path {step + 1}')

        ax_eta.set_title(f'η(x) After Solution {step + 1}',
                         fontsize=12, fontweight='bold')
        ax_eta.set_xlabel('x', fontsize=10)
        ax_eta.set_ylabel('y', fontsize=10)
        ax_eta.set_xlim(-2000, 2000)
        ax_eta.set_ylim(-2000, 2000)
        ax_eta.grid(True, linestyle='--', alpha=0.3)
        ax_eta.set_aspect('equal')
        ax_eta.legend(loc='upper right', fontsize=8)

        # Row 2: Deflated function
        ax_deflated = axes[2, step]
        Z_deflated_log = np.log1p(Z_deflated)
        contour_deflated = ax_deflated.contourf(X, Y, Z_deflated_log, levels=50,
                                                cmap='coolwarm', alpha=0.7)
        ax_deflated.contour(X, Y, Z_deflated_log, levels=20, colors='white',
                            alpha=0.3, linewidths=0.5)
        plt.colorbar(contour_deflated, ax=ax_deflated, label='log(1 + Cost × η)')

        ax_deflated.plot(path_array[:, 0], path_array[:, 1], 'o-',
                         label=f"Path {step + 1}", markersize=4, linewidth=2.5)

        ax_deflated.plot(x0[0], x0[1], 'kx', markersize=15,
                         markeredgewidth=3, label='Start', zorder=10)

        if solutions:
            sols_array = np.array(solutions)
            ax_deflated.scatter(sols_array[:, 0], sols_array[:, 1],
                                c='red', s=200, marker='*', edgecolors='white',
                                linewidths=2, label='Solutions', zorder=10)

        ax_deflated.set_title(f'Deflated Function (Step {step + 1})',
                              fontsize=12, fontweight='bold')
        ax_deflated.set_xlabel('x', fontsize=10)
        ax_deflated.set_ylabel('y', fontsize=10)
        ax_deflated.legend(loc='best', fontsize=8)
        ax_deflated.set_xlim(-2000, 2000)
        ax_deflated.set_ylim(-2000, 2000)
        ax_deflated.grid(True, linestyle='--', alpha=0.3)
        ax_deflated.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('acoustic_phase_residual_deflation.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to acoustic_phase_residual_deflation.png")
    plt.show()

    # Print summary
    print("\n=== Summary ===")
    print(f"Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i + 1}: {sol} (cost: {acoustic_cost(sol):.6e})")


def deflation_on_test_problem_with_many_local_minima():
    from functions import test_problem_with_many_local_minima, test_problem_with_many_local_minima_jacobian_scipy, \
        jacobian_test_problem_auto
    from good import good, make_deflation_funcs
    x0 = np.array([1.0, 3.0])
    sols = []
    for i in range(50):
        if len(sols) > 0:
            random_sol = sols[np.random.randint(len(sols))]
            # x0 = random_sol + np.random.randn(2) * 0.1
        mu1, grad_eta1 = make_deflation_funcs(sols)
        min1, path1 = good(test_problem_with_many_local_minima,
                           jacobian_test_problem_auto,
                           grad_eta1,
                           x0)
        sols.append(min1)
        print(f"min1: {min1}")

    for sol in sols:
        plt.plot(sol[0], sol[1], 'o-', label=f"sol {sol}")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


def find_minimas_with_scipy():
    from functions import test_problem_with_many_local_minima, test_problem_with_many_local_minima_jacobian_scipy
    from scipy.optimize import shgo, least_squares
    def scalar_objective(x):
        residuals = test_problem_with_many_local_minima(x)
        return 0.5 * np.sum(residuals ** 2)  # Sum of Squares (Energy)

    # --- 3. The Solver Routine ---

    bounds = [(-4, 4), (-4, 4)]
    print("Running Global Search...")
    result_global = shgo(scalar_objective, bounds, sampling_method='sobol', n=1000000)
    print("found minimas:", len(result_global.xl))
    print(f"Global optimization found", result_global.xl)
    for minima in result_global.xl:
        plt.plot(minima[0], minima[1] , 'o', label="minima")
    plt.show()


if __name__ == "__main__":
    # test_ackley_func()
    # deflation_on_test_problem_with_many_local_minima()
    find_minimas_with_scipy()
