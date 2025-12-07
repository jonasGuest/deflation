import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Callable
from scipy.optimize import approx_fprime
import jax
import jax.numpy as jnp

def himmelblau(xy: np.ndarray) -> np.ndarray:
    x, y = xy[0], xy[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def test_problem_with_many_local_minima(xy: np.ndarray, a: float = 10.0) -> np.ndarray:
    x, y = xy[0], xy[1]
    r1 = a
    r2 = a
    for k in range(1,4):
        r1 *= 1-(x+y)**2 / (k**2*np.pi**2)
        r2 *= 1 - (x-y)**2 / ((k-0.5)**2 *np.pi**2)
    r3 = a + 0.01 * (x**2 + y**2)
    return np.array([r1, r2, r3])

def test_problem_with_many_local_minima_jacobian_scipy(xy: np.ndarray, a: float = 10.0) -> np.ndarray:
    f = lambda x: test_problem_with_many_local_minima(x, a)
    # Compute each row of the Jacobian
    J = np.zeros((3, 2))
    for i in range(3):
        # Extract i-th component of residual
        f_i = lambda x: f(x)[i]
        J[i, :] = approx_fprime(xy, f_i, epsilon=1e-8)

    return J

# performance problem
def test_problem_jax(xy: jnp.ndarray, a: float = 10.0) -> jnp.ndarray:
    x, y = xy[0], xy[1]
    r1 = a
    r2 = a
    for k in range(1, 4):
        r1 *= 1 - (x + y)**2 / (k**2 * jnp.pi**2)
        r2 *= 1 - (x - y)**2 / ((k - 0.5)**2 * jnp.pi**2)
    r3 = a + 0.01 * (x**2 + y**2)
    return jnp.array([r1, r2, r3])

def jacobian_test_problem_auto(xy: np.ndarray, a: float = 10.0) -> np.ndarray:
    xy_jax = jnp.array(xy)

    # Compute Jacobian using JAX
    jacobian_fn = jax.jacfwd(lambda x: test_problem_jax(x, a))
    J_jax = jacobian_fn(xy_jax)

    # Convert back to NumPy
    return np.array(J_jax)

def residual_to_scalar(r: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], float]:
    def f(x: np.ndarray) -> float:
        res = r(x)
        return 0.5 * np.dot(res, res)
    return f

def visualize_himmelblau():
    # --- Himmelblau Visualization ---
    x_him = np.linspace(-6, 6, 1000)
    y_him = np.linspace(-6, 6, 1000)
    X_him, Y_him = np.meshgrid(x_him, y_him)
    Z_him = np.zeros_like(X_him)
    for i in range(X_him.shape[0]):
        for j in range(X_him.shape[1]):
            Z_him[i, j] = himmelblau(np.array([X_him[i, j], Y_him[i, j]]))

    fig_him = go.Figure(data=[
        go.Surface(z=Z_him, x=x_him, y=y_him, colorscale='Turbo', showscale=True, cmax=200, cmin=0)
    ])

    fig_him.update_layout(
        title_text="Himmelblau Function",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        height=700,
        width=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig_him.show()

def visualize_test_problem_with_many_local_minima():
    x_test = np.linspace(-2, 2, 250)
    y_test = np.linspace(-2, 5, 250)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    Z_test = np.zeros_like(X_test)
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            r = test_problem_with_many_local_minima(np.array([X_test[i, j], Y_test[i, j]]), a=4)
            Z_test[i, j] = 0.5 * np.dot(r, r)

    fig_test = go.Figure(data=[
        go.Surface(z=Z_test, x=x_test, y=y_test, colorscale='Viridis', showscale=True)
    ])

    fig_test.update_layout(
        title_text="Test Problem with Many Local Minima (Residual Norm)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (||residual||)',
            aspectmode='cube'
        ),
        height=700,
        width=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig_test.show()

if __name__ == "__main__":
    visualize_test_problem_with_many_local_minima()
