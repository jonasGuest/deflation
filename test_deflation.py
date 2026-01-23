from typing import Callable, Optional, Dict, Any, List

import good
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm


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


def deflation_on_test_problem_with_many_local_minima():
    from functions import test_problem_jax, jacobian_test_problem_auto
    from good import good, make_deflation_funcs
    x0 = np.array([1.0, 3.0])
    sols = []

    def residual_func(x):
        return np.array(test_problem_jax(x))
    def jacobian_func(x):
        return jacobian_test_problem_auto(x)
    for i in range(42):
        mu1, grad_eta1 = make_deflation_funcs(sols)
        min1, path1 = good(residual_func,
                           jacobian_func,
                           grad_eta1,
                           x0,
                           epsilon=0.01,
                           tol=1e-3,
                           max_iter=300,
                           limit_step_undeflated=2.0,
                           )
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


def plot_himmelblau():
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

def test_himmelblau():
    from good import make_deflation_funcs, good
    from functions import himmelblau_residual, himmelblau_jacobi

    epsilon = 0.01

    sols = []
    x0 = np.array([0, 0])
    for i in range(4):
        mu1, grad_eta1 = make_deflation_funcs(sols)
        min1, path1 = good(himmelblau_residual,
                           himmelblau_jacobi,
                           grad_eta1,
                           x0,
                           epsilon=epsilon,
                           tol=1e-3,
                           max_iter=300,
                           limit_step_undeflated=3.0,
                           )
        sols.append(min1)
        print(f"min1: {min1}")
    xs = np.linspace(-6, 6, 500)
    ys = np.linspace(-6, 6, 500)
    X, Y = np.meshgrid(xs, ys)
    beta_values = np.zeros(X.shape)

    _, grad_eta = make_deflation_funcs(sols[:3])

    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            xy = np.array([X[i, j], Y[i, j]])
            beta = good(himmelblau_residual, himmelblau_jacobi, grad_eta, xy, epsilon=epsilon, return_beta_instantly=True)
            beta_values[i, j] = beta
    plt.figure(figsize=(8, 6))
    cmap_colors = ['lightcoral', 'orange', 'green']
    cmap = ListedColormap(cmap_colors)

    bounds = [-1000, -epsilon, 1-epsilon, 1000]
    norm = BoundaryNorm(bounds, cmap.N)

    mesh = plt.pcolormesh(X, Y, beta_values, cmap=cmap, norm=norm, shading='auto')

    cbar = plt.colorbar(mesh, ticks=[-0.5, 0.5, 1.5])
    cbar.ax.set_yticklabels(['Beta < 0', '0 < Beta < 1', 'Beta > 1'])

    plt.contour(X, Y, (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2, levels=20, colors='k', alpha=0.1, linewidths=0.5)
    for idx, sol in enumerate(sols):
        plt.plot(sol[0], sol[1], marker='x', color='black', markersize=12, markeredgewidth=3, label=f'Sol {idx+1}')

    plt.title("Himmelblau Solutions & Beta Stability Regions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()


import matplotlib.patches as mpatches


def test_himmelblau_eps_zero():
    from good import make_deflation_funcs, good
    from functions import himmelblau_residual, himmelblau_jacobi

    epsilon = 0.00

    sols = []
    x0 = np.array([0, 0])
    for i in range(4):
        mu1, grad_eta1 = make_deflation_funcs(sols)
        min1, path1 = good(himmelblau_residual,
                           himmelblau_jacobi,
                           grad_eta1,
                           x0,
                           epsilon=epsilon,
                           tol=1e-3,
                           max_iter=300,
                           limit_step_undeflated=3.0,
                           )
        sols.append(min1)
        print(f"min1: {min1}")
    xs = np.linspace(-6, 6, 500)
    ys = np.linspace(-6, 6, 500)
    X, Y = np.meshgrid(xs, ys)
    beta_values = np.zeros(X.shape)

    _, grad_eta = make_deflation_funcs(sols[:3])

    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            xy = np.array([X[i, j], Y[i, j]])
            beta = good(himmelblau_residual, himmelblau_jacobi, grad_eta, xy, epsilon=epsilon, return_beta_instantly=True)
            beta_values[i, j] = beta
    plt.figure(figsize=(6, 4))
    cmap_colors = ['lightcoral', 'orange', 'green']
    cmap = ListedColormap(cmap_colors)

    bounds = [-1000, -epsilon, 1-epsilon, 1000]
    norm = BoundaryNorm(bounds, cmap.N)

    mesh = plt.pcolormesh(X, Y, beta_values, cmap=cmap, norm=norm, shading='auto')

    cbar = plt.colorbar(mesh, ticks=[-0.5, 0.5, 1.5])
    cbar.ax.set_yticklabels(['Beta < 0', '0 < Beta < 1', 'Beta > 1'])

    plt.contour(X, Y, (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2, levels=20, colors='k', alpha=0.1, linewidths=0.5)
    for idx, sol in enumerate(sols):
        plt.plot(sol[0], sol[1], marker='x', color='black', markersize=12, markeredgewidth=3, label=f'Sol {idx+1}')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()

def test_rosenbrock():
    from good import make_deflation_funcs, good
    from functions import rosenbrock_residual, rosenbrock_jacobi
    sols = []

    x0 = np.array([0, 0])
    mu1, grad_eta1 = make_deflation_funcs(sols)
    min1, path1 = good(rosenbrock_residual,
                       rosenbrock_jacobi,
                       grad_eta1,
                       x0,
                       epsilon=0.01,
                       tol=1e-3,
                       max_iter=300,
                       limit_step_undeflated=2.0,
                       )
    print(f"min1: {min1}")


if __name__ == "__main__":
    # test_ackley_func()
    deflation_on_test_problem_with_many_local_minima()
    # test_himmelblau()
    # test_himmelblau()
    # test_himmelblau_eps_zero()
    # find_minimas_with_scipy()
    # test_rosenbrock()
