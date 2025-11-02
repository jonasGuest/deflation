import numpy as np
import matplotlib.pyplot as plt


# --- 1. Define Model, Residual, and Jacobian ---

def model(x, beta):
    """The model function f(x, beta)"""
    # Our model is f(x, beta) = beta * x^2
    return beta * x ** 2


def residual(x, y, beta):
    """The residual function r_i(beta)"""
    # r_i = y_i - f(x_i, beta)
    return y - model(x, beta)


def jacobian(x):
    """The Jacobian J_i(beta) = d(f)/d(beta)"""
    # J_i = d/d(beta) [beta * x_i^2] = x_i^2
    # Note: In this specific case, the Jacobian doesn't depend on beta.
    return x ** 2


# --- 2. Generate Sample Data ---

# Let's create some data where the true beta is 3.0
beta_true = 3.0
num_points = 20

# Generate x data
x_data = np.linspace(-5, 5, num_points)

# Generate y data with some random noise
# y = 3.0 * x^2 + noise
np.random.seed(42)  # for reproducible results
y_noise = np.random.normal(0, 3.5, num_points)  # Add Gaussian noise
y_data = beta_true * x_data ** 2 + y_noise

# --- 3. Run the Gauss-Newton Algorithm ---

# Set algorithm parameters
iterations = 5
initial_guess = 0.5
beta_k = initial_guess

# Set up the plot
# We'll create a grid of subplots to show the progress
# Using (3, 2) grid for 5 iterations
plt.figure(figsize=(12, 15))
plt.suptitle(f"Gauss-Newton Optimization for $y = \\beta x^2$", fontsize=16, y=1.02)

for k in range(iterations):
    # --- Plotting Step ---
    ax = plt.subplot(3, 2, k + 1)

    # Plot the original data points
    ax.scatter(x_data, y_data, label=f'Data (True $\\beta$={beta_true})')

    # Plot the current model fit
    # Create a smooth x-axis for plotting the curve
    x_fit = np.linspace(-5, 5, 100)
    y_fit = model(x_fit, beta_k)

    ax.plot(x_fit, y_fit, 'r-', label=f'Current Fit ($\k={k}$)')
    ax.set_title(f"Iteration {k}: $\\beta_k$ = {beta_k:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Gauss-Newton Update Step ---

    # 1. Calculate residuals and Jacobian at the current beta_k
    r = residual(x_data, y_data, beta_k)
    J = jacobian(x_data)

    # 2. Calculate the terms for the update formula
    # Numerator: sum(J_i * r_i)
    numerator = np.sum(J * r)

    # Denominator: sum(J_i^2)
    denominator = np.sum(J ** 2)

    # 3. Calculate the update step delta_beta
    # Handle potential division by zero, though unlikely here
    if denominator == 0:
        print("Error: Denominator is zero. Cannot update.")
        break

    delta_beta = numerator / denominator

    # 4. Update beta_k for the next iteration
    beta_k = beta_k + delta_beta

# --- Final Results ---
print(f"Algorithm Finished.")
print(f"Initial Guess (beta_0): {initial_guess}")
print(f"True Beta:              {beta_true}")
print(f"Estimated Beta (beta_{iterations}): {beta_k:.6f}")

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()