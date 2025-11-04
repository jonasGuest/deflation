import numpy as np
import plotly.graph_objects as go
from good import residual_scalar_func, make_deflation_funcs


def plot3d(title, f, x_range, y_range):
    X, Y = np.meshgrid(x_range, y_range)

    # Compute scalar residual: 0.5 * ||r(x)||^2
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(pt)

    # Create Plotly surface
    surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', colorbar=dict(title='0.5||r||^2'))
    fig = go.Figure(data=[surface])

    # Layout and camera
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='0.5||r||^2',
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
    eta_func, grad_eta_func = make_deflation_funcs([solutions[0]])
    g = lambda xy: residual_scalar_func(xy) * eta_func(xy)
    # Grid setup
    N = 200  # increase/decrease for resolution vs speed
    x_range = np.linspace(-4.5, 4.5, N)
    y_range = np.linspace(-4.5, 4.5, N)
    plot3d('f', g, x_range, y_range)
    plot3d('eta_func', eta_func, x_range, y_range)
    plot3d('residual_scalar_func', residual_scalar_func, x_range, y_range)

