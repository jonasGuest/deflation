import numpy as np
import matplotlib.pyplot as plt
from good import good, make_deflation_funcs
from utils import numerical_jacobian

np.random.seed(20)

def steering_vector(theta_deg, M, d_lambda=0.5):
    """
    Calculates the steering vector for a ULA.
    theta_deg: Angle in degrees (0 is broadside).
    M: Number of antenna elements.
    d_lambda: Element spacing in wavelengths.
    """
    theta_rad = np.deg2rad(theta_deg)
    # Phase shift exponent: -j * 2*pi * d * sin(theta) * m
    # indices m = 0, ..., M-1
    m = np.arange(M)
    # Using the standard physics convention: exp(-j * k * x)
    # a(theta) = [1, exp(-j*phi), ..., exp(-j*(M-1)phi)]^T
    # where phi = 2*pi * d * sin(theta)
    phi = 2 * np.pi * d_lambda * np.sin(theta_rad)
    a = np.exp(-1j * m * phi)
    return a


def steering_derivative(theta_deg, M, d_lambda=0.5):
    """
    Calculates the derivative of the steering vector with respect to theta.
    da/dtheta
    """
    theta_rad = np.deg2rad(theta_deg)
    m = np.arange(M)
    phi = 2 * np.pi * d_lambda * np.sin(theta_rad)

    # a_m = exp(-j * m * 2*pi * d * sin(theta))
    # da_m/dtheta = a_m * (-j * m * 2*pi * d * cos(theta))

    derivative_factor = -1j * m * 2 * np.pi * d_lambda * np.cos(theta_rad)
    a = np.exp(-1j * m * phi)
    da = a * derivative_factor
    return da


def simulate_data(M, N, true_angle, snr_db):
    """
    Generates synthetic data X (M x N).
    """
    # Signal
    t = np.arange(N)
    # Simple complex sine wave signal
    signal = np.exp(1j * 2 * np.pi * 0.1 * t)

    # Array response
    a = steering_vector(true_angle, M)
    # X_signal = a * s(t) -> shape (M, N)
    X_sig = np.outer(a, signal)

    # Noise
    noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

    # Scale signal for SNR
    # SNR = 10 log10(Ps/Pn). Ps=1, Pn=1 (normalized).
    # We scale the signal amplitude A. A^2 / 1 = 10^(SNR/10)
    amp = np.sqrt(10 ** (snr_db / 10))

    X = amp * X_sig + noise
    return X


def capon_lm_1d(X, initial_guess, max_iter=50, lambda_param=0.01):
    """
    Implements the Levenberg-Marquardt algorithm for Capon 1D.

    Cost Function: F(theta) = || X^H * (XX^H)^-1 * a(theta) ||^2
    This simplifies to calculating residuals v = C * a(theta), where C = X^H (XX^H)^-1.
    """
    M, N = X.shape

    # 1. Precompute fixed matrices
    R_hat = (X @ X.conj().T) / N
    R_inv = np.linalg.inv(R_hat)

    # To match the "Least Squares" form F = ||v||^2
    # We derived v = X^H * R_hat^-1 * a(theta) / N  (approx scaling)
    # Actually, simpler: F = a^H R^-1 a.
    # We need a vector v such that v^H v = a^H R^-1 a.
    # Decompose R^-1 = L L^H (Cholesky). Then v = L^H a.
    # This creates a vector of size M (residuals).
    # This is computationally cheaper and mathematically equivalent for the optimizer.

    try:
        L = np.linalg.cholesky(R_inv)  # R_inv = L * L.H
        # F = a^H L L^H a = || L^H a ||^2
        # Let residual vector r(theta) = L^H a(theta) (Complex vector size M)
        # We treat real and imag parts as separate residuals for LM (Size 2M).
        precomputed_matrix = L.conj().T
    except np.linalg.LinAlgError:
        print("Matrix not positive definite, adding regularization...")
        R_inv_reg = R_inv + np.eye(M) * 1e-6
        L = np.linalg.cholesky(R_inv_reg)
        precomputed_matrix = L.conj().T

    theta = initial_guess
    history = [theta]

    # LM Loop
    mu = lambda_param  # Damping factor

    for i in range(max_iter):
        # Current Vector and Derivative
        a = steering_vector(theta, M)
        da = steering_derivative(theta, M)

        # Calculate Residual Vector r (complex)
        # r = L^H * a
        r_complex = precomputed_matrix @ a

        # Calculate Jacobian columns associated with r (complex)
        # dr/dtheta = L^H * da
        J_complex = precomputed_matrix @ da

        # Convert to Real format for standard LM implementation
        # Residuals vector f = [Re(r); Im(r)]  (Size 2M x 1)
        f = np.concatenate([np.real(r_complex), np.imag(r_complex)])

        # Jacobian Matrix J (Size 2M x 1)
        # J = [Re(dr/dtheta); Im(dr/dtheta)]
        J = np.concatenate([np.real(J_complex), np.imag(J_complex)])

        # Reshape J to be a column vector (matrix) for matmul
        J = J.reshape(-1, 1)

        # --- LM Update Step ---
        # delta = - (J.T @ J + mu * I)^-1 @ J.T @ f

        H = J.T @ J  # Approximate Hessian (Scalar in 1D)
        g = J.T @ f  # Gradient (Scalar)

        # Update logic
        # For 1D, matrix inversion is just division
        delta = - g / (H + mu)

        theta_new = theta + delta.item()

        # Check if cost decreased
        a_new = steering_vector(theta_new, M)
        r_new = precomputed_matrix @ a_new
        cost_old = np.sum(np.abs(r_complex) ** 2)
        cost_new = np.sum(np.abs(r_new) ** 2)

        if cost_new < cost_old:
            theta = theta_new
            mu /= 10  # Reduce damping (more like Gauss-Newton)
            history.append(theta)
            # Stopping criteria (small step)
            if np.abs(delta) < 1e-6:
                break
        else:
            mu *= 10  # Increase damping (more like Gradient Descent)
            # Do not update theta

    return theta, history


# --- Main Execution ---

# ... (Keep the previous function definitions: steering_vector, steering_derivative, etc.) ...

def get_coarse_estimate(X, M, step=4.0):
    """
    Performs a coarse grid search to find a good initial guess for LM.
    As suggested in the paper[cite: 175, 192].
    """
    # Create a coarse grid (e.g., every 4 degrees as per paper source 193)
    grid = np.arange(-90, 90 + step, step)
    costs = []

    # Precompute R_inv once
    N = X.shape[1]
    R_hat = (X @ X.conj().T) / N
    R_inv = np.linalg.inv(R_hat)

    # Evaluate cost at each coarse point
    for theta in grid:
        a = steering_vector(theta, M)
        # Cost F = a^H R^-1 a
        cost = np.real(a.conj().T @ R_inv @ a)
        costs.append(cost)

    # Find the angle with the minimum cost
    min_idx = np.argmin(costs)
    return grid[min_idx]

# --- Improved Main Execution ---

def get_residual_vector(theta_deg, L_H, M, d_lambda=0.5):
    """
    Computes the explicit residual vector for the Capon LM algorithm.

    Parameters:
        theta_deg (float): The current angle guess in degrees.
        L_H (ndarray): The Hermitian transpose of the Cholesky factor of R_inv.
                       Shape (M, M). Precomputed as: L_H = np.linalg.cholesky(R_inv).conj().T
        M (int): Number of antenna elements.
        d_lambda (float): Spacing in wavelengths (default 0.5).

    Returns:
        residuals (ndarray): A real-valued vector of size 2*M.
                             [Re(r_1)...Re(r_M), Im(r_1)...Im(r_M)]
    """
    # 1. Calculate Steering Vector a(theta)
    theta_rad = np.deg2rad(theta_deg)
    m_indices = np.arange(M)
    phi = 2 * np.pi * d_lambda * np.sin(theta_rad)
    a = np.exp(-1j * m_indices * phi)

    # 2. Calculate Complex Residuals v = L^H * a
    # This vector v satisfies: ||v||^2 = a^H * R^-1 * a
    v_complex = L_H @ a

    # 3. Stack Real and Imaginary parts
    # LM solvers require real residuals. We treat Re and Im as independent data points.
    residuals = np.concatenate([np.real(v_complex), np.imag(v_complex)])

    return residuals

# Parameters
M = 8
N = 100
true_aoa = 25.0
snr = 10

# 1. Generate Data
X = simulate_data(M, N, true_aoa, snr)

# 2. Step A: Coarse Search (The Paper's Technique)
# We scan roughly to avoid getting stuck in sidelobes
coarse_guess = get_coarse_estimate(X, M, step=5.0)
print(f"Coarse Estimate: {coarse_guess}°")

# 3. Step B: Fine Tuning with LM
# Now we pass the coarse guess to LM instead of a random number
estimated_aoa, path = capon_lm_1d(X, coarse_guess)

print(f"Refined Estimate (LM): {estimated_aoa:.4f}°")
print(f"True Angle: {true_aoa}°")

# --- Visualization ---
angles = np.linspace(-90, 90, 360)
R_inv = np.linalg.inv((X @ X.conj().T) / N)
costs = [np.real(steering_vector(a, M).conj().T @ R_inv @ steering_vector(a, M)) for a in angles]

plt.figure(figsize=(10, 6))
plt.plot(angles, costs, label='Cost Function')
plt.plot(path, [np.real(steering_vector(p, M).conj().T @ R_inv @ steering_vector(p, M)) for p in path],
         'r-o', label='LM Optimization Path')
plt.axvline(true_aoa, color='green', linestyle=':', label='True AoA')
plt.title('Corrected: Coarse Search + Levenberg-Marquardt')
plt.xlabel('Angle')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

from scipy.optimize import least_squares

# Precompute L_H once
R_hat = (X @ X.conj().T) / N
R_inv = np.linalg.inv(R_hat)
L_H = np.linalg.cholesky(R_inv).conj().T

residual_func = lambda x: get_residual_vector(x[0], L_H, M, d_lambda=0.5)
jacobian = lambda x: numerical_jacobian(residual_func, x)
good_sols = []

for i in range(10):
    _, grad_eta = make_deflation_funcs(good_sols)
    good_sol, path = good(residual_func, jacobian, grad_eta, np.array([39.0]), tol=1e-6)
    print("good_sol: ", good_sol)

    good_sols.append(good_sol)
    plt.axvline(good_sol[0], linestyle=':', label=f'least_squares_{i}')
plt.show()
