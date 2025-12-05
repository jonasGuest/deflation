import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple

def rastrigin(xy: np.ndarray, A: float = 10) -> np.ndarray:
    """
    Rastrigin function.
    Global minimum at (0, 0) with value 0.

    Args:
        xy: numpy array of shape (2,) with [x, y] coordinates
        A: amplitude parameter (default 10)

    Returns:
        Function value at xy
    """
    x, y = xy[0], xy[1]
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def ackley(xy: np.ndarray) -> float:
    """
    Ackley function.
    Global minimum at (0, 0) with value 0.

    Args:
        xy: numpy array of shape (2,) with [x, y] coordinates

    Returns:
        Function value at xy
    """
    x, y = xy[0], xy[1]
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - \
            np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + \
            np.e + 20

def himmelblau(xy: np.ndarray) -> np.ndarray:
    """
    Himmelblau function.
    Has 4 identical global minima.

    Args:
        xy: numpy array of shape (2,) with [x, y] coordinates

    Returns:
        Function value at xy
    """
    x, y = xy[0], xy[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def eggholder(xy: np.ndarray) -> np.ndarray:
    """
    Eggholder function.
    Global minimum is typically at (512, 404.2319).

    Args:
        xy: numpy array of shape (2,) with [x, y] coordinates

    Returns:
        Function value at xy
    """
    x, y = xy[0], xy[1]
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47))))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2


if __name__ == "__main__":
    # Rastrigin Data
    x_rast = np.linspace(-5.12, 5.12, 100)
    y_rast = np.linspace(-5.12, 5.12, 100)
    X_rast, Y_rast = np.meshgrid(x_rast, y_rast)
    Z_rast = np.zeros_like(X_rast)
    for i in range(X_rast.shape[0]):
        for j in range(X_rast.shape[1]):
            Z_rast[i, j] = rastrigin(np.array([X_rast[i, j], Y_rast[i, j]]))

    # Ackley Data
    x_ack = np.linspace(-5, 5, 100)
    y_ack = np.linspace(-5, 5, 100)
    X_ack, Y_ack = np.meshgrid(x_ack, y_ack)
    Z_ack = np.zeros_like(X_ack)
    for i in range(X_ack.shape[0]):
        for j in range(X_ack.shape[1]):
            Z_ack[i, j] = ackley(np.array([X_ack[i, j], Y_ack[i, j]]))

    # Himmelblau Data
    x_him = np.linspace(-6, 6, 100)
    y_him = np.linspace(-6, 6, 100)
    X_him, Y_him = np.meshgrid(x_him, y_him)
    Z_him = np.zeros_like(X_him)
    for i in range(X_him.shape[0]):
        for j in range(X_him.shape[1]):
            Z_him[i, j] = himmelblau(np.array([X_him[i, j], Y_him[i, j]]))

    # Eggholder Data (Needs a larger range)
    x_egg = np.linspace(-512, 512, 100)
    y_egg = np.linspace(-512, 512, 100)
    X_egg, Y_egg = np.meshgrid(x_egg, y_egg)
    Z_egg = np.zeros_like(X_egg)
    for i in range(X_egg.shape[0]):
        for j in range(X_egg.shape[1]):
            Z_egg[i, j] = eggholder(np.array([X_egg[i, j], Y_egg[i, j]]))

    # --- 3. Create Subplots ---

    # We must specify 'type': 'surface' for 3D plots in the specs
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Rastrigin Function', 'Ackley Function',
                        'Himmelblau Function', 'Eggholder Function'),
        horizontal_spacing=0.03,  # reduce horizontal gap (default ~0.2)
        vertical_spacing=0.08     # reduce vertical gap (default ~0.3)
    )

    # Add Rastrigin
    fig.add_trace(
        go.Surface(z=Z_rast, x=x_rast, y=y_rast, colorscale='Viridis', showscale=False),
        row=1, col=1
    )

    # Add Ackley
    fig.add_trace(
        go.Surface(z=Z_ack, x=x_ack, y=y_ack, colorscale='Plasma', showscale=False),
        row=1, col=2
    )

    # Add Himmelblau
    # We clamp the visual range slightly to see the minima better using cmin/cmax
    fig.add_trace(
        go.Surface(z=Z_him, x=x_him, y=y_him, colorscale='Turbo', showscale=False, cmax=200, cmin=0),
        row=2, col=1
    )

    # Add Eggholder
    fig.add_trace(
        go.Surface(z=Z_egg, x=x_egg, y=y_egg, colorscale='Earth', showscale=False),
        row=2, col=2
    )

    # --- 4. Final Layout Adjustments ---

    fig.update_layout(
        title_text="3D Optimization Test Functions (Interactive)",
        height=700,
        width=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Fix aspect ratios and axis titles for each subplot scene
    fig.update_scenes(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'  # Ensures the plot doesn't look flattened
    )

    fig.show()


def acoustic_phase_residual(xy: np.ndarray, a: float = 1.0) -> np.ndarray:
    """
    Acoustic phase localization residual function with regularization.

    This function has multiple minima due to spatial aliasing in acoustic arrays.
    The residual includes:
    - Phase matching terms for different harmonics k=1,2,3
    - Regularization term penalizing distance from origin

    Args:
        xy: numpy array of shape (2,) with [x, y] coordinates
        a: amplitude parameter (default 1.0)

    Returns:
        Residual vector of shape (7,) - 6 phase terms + 1 regularization term
    """
    x, y = xy[0], xy[1]
    r_squared = x ** 2 + y ** 2

    # Initialize residual vector
    residual = np.zeros(7)

    # First set of terms: a * ∏(1 - (x+y)²/(k²π²)) for k=1,2,3
    for idx, k in enumerate([1, 2, 3]):
        residual[idx] = a * (1 - (x + y) ** 2 / (k ** 2 * np.pi ** 2))

    # Second set of terms: a * ∏(1 - (x-y)²/((k-1/2)²π²)) for k=1,2,3
    for idx, k in enumerate([1, 2, 3]):
        k_shifted = k - 0.5
        residual[idx + 3] = a * (1 - (x - y) ** 2 / (k_shifted ** 2 * np.pi ** 2))

    # Regularization term
    residual[6] = a + 0.01 * r_squared

    return residual