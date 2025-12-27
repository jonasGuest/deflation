import numpy as np
import matplotlib.pyplot as plt

# Assume these exist from your previous files
from good import good, make_deflation_funcs
from utils import numerical_jacobian


def generate_angle_data():
    """
    Generates data for a source at a specific angle (Far-Field / Plane Wave).
    """
    c = 343.0
    f = 2000.0
    wavelength = c / f

    # Linear array: 5 mics, 20cm spacing along X-axis
    # Spacing (0.2) > Wavelength (0.17) ==> Aliasing will occur
    mics = np.zeros((5, 2))
    mics[:, 0] = np.linspace(0, 0.8, 5)

    # True Direction: 60 degrees
    true_angle_rad = np.radians(60)

    # Direction Vector (pointing TO the source)
    # This is the normal vector of the wavefront
    dir_vec = np.array([np.cos(true_angle_rad), np.sin(true_angle_rad)])

    # Calculate distances based on projection of mic position onto direction vector
    # Equation: distance = dot(mic_pos, direction_vector)
    projected_dists = np.dot(mics, dir_vec)

    # Phases relative to the first mic (index 0)
    dist_diffs = projected_dists - projected_dists[0]
    measured_phases = (2 * np.pi * dist_diffs / wavelength) % (2 * np.pi)

    return mics, measured_phases, wavelength, true_angle_rad


def angle_residual(theta, mics, measured_phases, wavelength):
    """
    Residual function for a single angle theta (1D input).
    """
    # 1. Unpack angle (theta is a 1-element array coming from the optimizer)
    t = theta[0]

    # 2. Create direction vector
    dir_vec = np.array([np.cos(t), np.sin(t)])

    # 3. Model: Project mics onto this direction (Plane Wave assumption)
    projected_dists = np.dot(mics, dir_vec)
    dist_diffs = projected_dists - projected_dists[0]
    model_phases = 2 * np.pi * dist_diffs / wavelength

    # 4. Residuals (Unit circle distance)
    phase_diff = measured_phases - measured_phases[0]

    # We define error as the vector difference in the complex plane
    # (Real error, Imag error) for each mic
    r_cos = np.cos(phase_diff) - np.cos(model_phases)
    r_sin = np.sin(phase_diff) - np.sin(model_phases)

    return np.concatenate([r_cos, r_sin])


def solve_angles():
    # 1. Setup
    mics, measured_phases, wavelength, true_angle = generate_angle_data()

    # Define the callable functions for 'good'
    # 'good' expects f(x) and J(x)
    r_func = lambda x: angle_residual(x, mics, measured_phases, wavelength)
    J_func = lambda x: numerical_jacobian(r_func, x)

    solutions = []

    print(f"True Angle: {np.degrees(true_angle):.1f}°")

    # 2. Find Multiple Solutions (Deflation)
    # We try to find 3 solutions because we expect aliasing
    for i in range(3):
        # Create deflation wrapper based on previous solutions
        eta_func, grad_eta_func = make_deflation_funcs(solutions)

        # Start guess: 90 degrees (straight up)
        initial_guess = np.array([np.radians(90.0)])

        # Call the existing optimizer
        sol, path = good(
            r_func,
            J_func,
            grad_eta_func,
            initial_guess,
            epsilon=0.1,  # Shift size for deflation
            max_iter=50,
            verbose=False
        )

        # Store solution (normalize to 0-360 for readability)
        sol_scalar = sol[0] % (2 * np.pi)
        solutions.append(np.array([sol_scalar]))

        cost = 0.5 * np.sum(r_func(sol) ** 2)
        print(f"Found Solution {i + 1}: {np.degrees(sol_scalar):.1f}° (Cost: {cost:.2e})")

    # 3. Visualize
    visualize_1d(r_func, solutions, true_angle)


def visualize_1d(r_func, solutions, true_angle):
    """Plot the 1D cost function and the found minima."""
    angles = np.linspace(0, 180, 1000)
    costs = []

    for a in angles:
        rad = np.radians(a)
        r = r_func(np.array([rad]))
        costs.append(0.5 * np.sum(r ** 2))

    plt.figure(figsize=(10, 5))

    # Plot Landscape
    plt.plot(angles, costs, 'k-', alpha=0.6, label='Cost Function')

    # Plot True Angle
    plt.axvline(np.degrees(true_angle), color='g', linestyle='--', label='True Angle')

    # Plot Found Solutions
    for i, sol in enumerate(solutions):
        deg = np.degrees(sol[0])
        # Wrap for plotting if needed
        if deg > 180: deg -= 360
        plt.plot(deg, 0, 'r*', markersize=12, label=f'Found Sol {i + 1}' if i == 0 else "")

    plt.title("1D Angle Optimization (Plane Wave Model)")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Cost")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    solve_angles()