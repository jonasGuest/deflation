# python
import numpy as np
import plotly.graph_objects as go
from good import residual_scalar_func, make_deflation_funcs


def plot3d(title, f, x_range, y_range, overlay_disk=None):
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


if __name__ == "__main__":
    solutions = [np.array([2.99999999, 1.99999999]), np.array([-3.77931025, -3.28318599]), np.array([3.58442834, -1.84812653]),
                 np.array([-2.80511807, 3.13131252])]
    mu, grad_eta = make_deflation_funcs([solutions[0]])
    g = lambda xy: residual_scalar_func(xy) * mu(xy)

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
    plot3d('residual_scalar_func', residual_scalar_func, x_range, y_range)

    print(mu(np.array([3,1])), residual_scalar_func(np.array([3,1])), residual_scalar_func(np.array([3,1]))*mu(np.array([3,1])))
    print(g(np.array([3,1])))
    print(mu(np.array([3.0997,1.786])), residual_scalar_func(np.array([3.0997,1.786])), residual_scalar_func(np.array([3.0997,1.786]))*mu(np.array([3.0997,1.786])))
    print(g(np.array([3.0997,1.786])))