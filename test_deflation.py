from typing import Callable, Optional, Dict, Any, List

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


def plot3d(
        title: str,
        f: Callable[[np.ndarray], float],
        x_range: np.ndarray,
        y_range: np.ndarray,
        overlay_disk: Optional[Dict[str, Any]] = None
) -> None:
    import plotly.graph_objects as go
    """Plot a 3D surface of a scalar function.

    Args:
        title: Plot title
        f: Function that takes numpy array of shape (2,) and returns a scalar
        x_range: 1D numpy array of x coordinates
        y_range: 1D numpy array of y coordinates
        overlay_disk: Optional dictionary with disk overlay parameters
    """
    X, Y = np.meshgrid(x_range, y_range)

    # Compute scalar residual: 0.5 * ||r(x)||^2
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(pt)

    # Create Plotly surface
    surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')
    fig = go.Figure(data=[surface])

    # Optionally add a flat disk overlay (triangulated fan)
    if overlay_disk is not None:
        cx, cy = overlay_disk['center']
        radius = overlay_disk['radius']
        z_plane = overlay_disk.get('z', np.min(Z))  # default to min surface z
        color = overlay_disk.get('color', 'red')
        opacity = overlay_disk.get('opacity', 0.4)
        n_boundary = overlay_disk.get('n_boundary', 64)

        theta = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
        bx = cx + radius * np.cos(theta)
        by = cy + radius * np.sin(theta)
        bz = np.full_like(bx, z_plane)

        # vertices: center first, then boundary points
        x_verts = np.concatenate(([cx], bx))
        y_verts = np.concatenate(([cy], by))
        z_verts = np.concatenate(([z_plane], bz))

        # build triangle indices as a fan from the center
        i_idx = []
        j_idx = []
        k_idx = []
        for t in range(1, n_boundary):
            i_idx.append(0)
            j_idx.append(t)
            k_idx.append(t + 1)
        # close last triangle
        i_idx.append(0)
        j_idx.append(n_boundary)
        k_idx.append(1)

        mesh = go.Mesh3d(
            x=x_verts, y=y_verts, z=z_verts,
            i=i_idx, j=j_idx, k=k_idx,
            color=color, opacity=opacity, flatshading=True
        )
        fig.add_trace(mesh)

    # Layout and camera
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            aspectmode='auto',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.0))
        ),
        width=900,
        height=700
    )

    fig.show()


def test_himmelblau():
    solutions: List[np.ndarray] = [
        np.array([2.99999999, 1.99999999]),
        np.array([-3.77931025, -3.28318599]),
        np.array([3.58442834, -1.84812653]),
        np.array([-2.80511807, 3.13131252])
    ]
    mu, grad_eta = good.make_deflation_funcs([solutions[0]])
    g: Callable[[np.ndarray], float] = lambda xy: good.residual_scalar_func(xy) * mu(xy)

    # Grid setup
    N = 200
    x_range = np.linspace(-4.5, 4.5, N)
    y_range = np.linspace(-4.5, 4.5, N)

    # Build mesh and evaluate mu on grid to find mask where mu > 10
    Xg, Yg = np.meshgrid(x_range, y_range)
    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    mu_vals = np.array([mu(pt) for pt in pts])
    mask = mu_vals > 10

    masked_pts = pts[mask]
    # centroid and radius (max distance from centroid)
    center = masked_pts.mean(axis=0)
    radius = np.linalg.norm(masked_pts - center, axis=1).max()
    overlay = {
        'center': (float(center[0]), float(center[1])),
        'radius': float(radius),
        'z': 0.0,            # place disk on z=0 plane (adjust if needed)
        'color': 'red',
        'opacity': 0.35,
        'n_boundary': 128
    }

    plot3d('f', g, x_range, y_range, overlay_disk=overlay)
    plot3d('mu', lambda x: np.minimum(mu(x), 100), x_range, y_range)
    plot3d('residual_scalar_func', good.residual_scalar_func, x_range, y_range)

    print(mu(np.array([3,1])), good.residual_scalar_func(np.array([3, 1])), good.residual_scalar_func(np.array([3, 1])) * mu(np.array([3, 1])))
    print(g(np.array([3,1])))
    print(mu(np.array([3.0997,1.786])), good.residual_scalar_func(np.array([3.0997, 1.786])), good.residual_scalar_func(np.array([3.0997, 1.786])) * mu(np.array([3.0997, 1.786])))
    print(g(np.array([3.0997,1.786])))

if __name__ == "__main__":
    # test_ackley_func()
    deflation_on_test_problem_with_many_local_minima()
    # test_himmelblau()
    # find_minimas_with_scipy()
