import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Callable
from scipy.optimize import approx_fprime
import jax
import jax.numpy as jnp

def himmelblau(xy: np.ndarray) -> np.ndarray:
    x, y = xy[0], xy[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


import numpy as np

# based on https://github.com/AlbanBloorRiley/DeflatedGaussNewton/blob/615c185f3d61f044eab7cfd27f97f698ff341d8f/PaperFigures.m
def FTrig(xy, order=0):
    """
    Python translation of the MATLAB FTrig function.

    Parameters:
    xy : np.ndarray
        Input vector of shape (2,).
    order : int
        Determines what to return (similar to nargout):
        0: F (Scalar objective)
        1: F, R (Scalar, Residual vector)
        2: F, R, J (Scalar, Residual, Jacobian)
        3: F, R, J, H (Scalar, Residual, Jacobian, Hessian)

    Returns:
    Depends on 'order'.
    F: float
    R: np.ndarray (shape 3)
    J: np.ndarray (shape 3,2)
    H: np.ndarray (shape 2,2,3)
    """
    x = xy[0]
    y = xy[1]

    # --- R (Residuals) ---
    # Note: These numbers are hardcoded approximations of pi terms from the original text.

    r1 = -10 * (x + y) * \
         ((35184372088832 * (x + y) ** 2) / 3125302502557517 - 1) * \
         ((70368744177664 * (x + y) ** 2) / 2778046668940015 - 1) * \
         ((281474976710656 * (x + y) ** 2) / 2778046668940015 - 1)

    r2 = -10 * \
         ((1125899906842624 * (x - y) ** 2) / 2778046668940015 - 1) * \
         ((140737488355328 * (x - y) ** 2) / 8681395840437547 - 1) * \
         ((140737488355328 * (x - y) ** 2) / 3125302502557517 - 1)

    r3 = x ** 2 / 100 + y ** 2 / 100 + 10

    R = np.array([r1, r2, r3])
    F = np.sum(R ** 2)

    if order == 0:
        return F
    if order == 1:
        return F, R

    # --- J (Jacobian) ---
    if order >= 2:
        # The Jacobian derivatives (hardcoded from text)
        J = np.zeros((3, 2))

        # dR1/dx and dR1/dy (identical due to x+y term)
        # Note: MATLAB text combines these terms into a single massive line.
        # I have preserved the arithmetic exactly as written.
        j_r1 = -10 * ((35184372088832 * (x + y) ** 2) / 3125302502557517 - 1) * \
               ((70368744177664 * (x + y) ** 2) / 2778046668940015 - 1) * \
               ((281474976710656 * (x + y) ** 2) / 2778046668940015 - 1) - \
               10 * (x + y) * ((70368744177664 * x) / 3125302502557517 + (70368744177664 * y) / 3125302502557517) * \
               ((70368744177664 * (x + y) ** 2) / 2778046668940015 - 1) * \
               ((281474976710656 * (x + y) ** 2) / 2778046668940015 - 1) - \
               10 * (x + y) * ((35184372088832 * (x + y) ** 2) / 3125302502557517 - 1) * \
               ((70368744177664 * (x + y) ** 2) / 2778046668940015 - 1) * \
               ((562949953421312 * x) / 2778046668940015 + (562949953421312 * y) / 2778046668940015) - \
               10 * (x + y) * ((35184372088832 * (x + y) ** 2) / 3125302502557517 - 1) * \
               ((281474976710656 * (x + y) ** 2) / 2778046668940015 - 1) * \
               ((140737488355328 * x) / 2778046668940015 + (140737488355328 * y) / 2778046668940015)

        J[0, 0] = j_r1
        J[0, 1] = j_r1

        # dR2/dx (Term 2)
        J[1, 0] = -10 * ((140737488355328 * (x - y) ** 2) / 8681395840437547 - 1) * \
                  ((2251799813685248 * x) / 2778046668940015 - (2251799813685248 * y) / 2778046668940015) * \
                  ((140737488355328 * (x - y) ** 2) / 3125302502557517 - 1) - \
                  10 * ((281474976710656 * x) / 8681395840437547 - (281474976710656 * y) / 8681395840437547) * \
                  ((1125899906842624 * (x - y) ** 2) / 2778046668940015 - 1) * \
                  ((140737488355328 * (x - y) ** 2) / 3125302502557517 - 1) - \
                  10 * ((1125899906842624 * (x - y) ** 2) / 2778046668940015 - 1) * \
                  ((281474976710656 * x) / 3125302502557517 - (281474976710656 * y) / 3125302502557517) * \
                  ((140737488355328 * (x - y) ** 2) / 8681395840437547 - 1)

        # dR2/dy (Term 2 inverse)
        # In the text, this is nearly identical to J[1,0] but signs flip on the (x-y) chain rule parts.
        J[1, 1] = 10 * ((140737488355328 * (x - y) ** 2) / 8681395840437547 - 1) * \
                  ((2251799813685248 * x) / 2778046668940015 - (2251799813685248 * y) / 2778046668940015) * \
                  ((140737488355328 * (x - y) ** 2) / 3125302502557517 - 1) + \
                  10 * ((281474976710656 * x) / 8681395840437547 - (281474976710656 * y) / 8681395840437547) * \
                  ((1125899906842624 * (x - y) ** 2) / 2778046668940015 - 1) * \
                  ((140737488355328 * (x - y) ** 2) / 3125302502557517 - 1) + \
                  10 * ((1125899906842624 * (x - y) ** 2) / 2778046668940015 - 1) * \
                  ((281474976710656 * x) / 3125302502557517 - (281474976710656 * y) / 3125302502557517) * \
                  ((140737488355328 * (x - y) ** 2) / 8681395840437547 - 1)

        # dR3 derivatives
        J[2, 0] = x / 50
        J[2, 1] = y / 50

        if order == 2:
            return F, R, J


def test_problem_with_many_local_minima(xy: np.ndarray, a: float = 10.0) -> np.ndarray:
    x, y = xy[0], xy[1]
    r1 = a * (x+y)
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

def test_problem_jax(xy: jnp.ndarray, a: float = 10.0) -> jnp.ndarray:
    x, y = xy[0], xy[1]
    r1 = a * (x+y)
    r2 = a
    for k in range(1, 4):
        r1 *= 1 - (x + y)**2 / (k**2 * jnp.pi**2)
        r2 *= 1 - (x - y)**2 / ((k - 0.5)**2 * jnp.pi**2)
    r3 = a + 0.01 * (x**2 + y**2)
    return jnp.array([r1, r2, r3])

def jacobian_test_problem_auto(xy: np.ndarray, a: float = 10.0) -> np.ndarray:
    xy_jax = jnp.array(xy)

    jacobian_fn = jax.jacfwd(lambda x: test_problem_jax(x, a))
    J_jax = jacobian_fn(xy_jax)

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
