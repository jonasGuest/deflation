import numpy as np
import matplotlib.pyplot as plt
from phase import generate_perfect_data, phase_residual
from good import good, make_deflation_funcs
from utils import numerical_jacobian

def solve_acoustic_angle(
        num_solutions: int = 1,
        initial_guess_angle: float = None, # Now a scalar (radians)
        visualize: bool = True
) -> list:
    """
    Find acoustic source directions (angles) using deflated Gauss-Newton.

    Args:
        num_solutions: Number of distinct angles to find
        initial_guess_angle: Starting angle in radians (default: pi/2)
        visualize: Whether to plot the 1D cost landscape
    """
    # 1. Setup Data
    mics, measured_phases, wavelength, true_source = generate_perfect_data()

    # Calculate true angle for reference
    true_angle = np.arctan2(true_source[1], true_source[0])

    print(f"=== Acoustic Angle Localization ===")
    print(f"True Source Angle: {np.degrees(true_angle):.1f}°")

    # Default guess: 90 degrees (straight ahead)
    if initial_guess_angle is None:
        initial_guess_angle = np.array([np.pi / 2])
    else:
        initial_guess_angle = np.array([initial_guess_angle])

    # 2. Define Projection (1D Angle -> 2D Position)
    # We assume a fixed large distance (Far-Field assumption) to calculate phase
    ASSUMED_DISTANCE = 100.0

    def get_pos_from_theta(theta_arr):
        theta = theta_arr[0]
        return np.array([
            ASSUMED_DISTANCE * np.cos(theta),
            ASSUMED_DISTANCE * np.sin(theta)
        ])

    # 3. Define Residuals with respect to Theta
    def r_func(theta_arr: np.ndarray) -> np.ndarray:
        # Map theta -> (x,y) -> residuals
        pos = get_pos_from_theta(theta_arr)
        return phase_residual(pos, mics, measured_phases, wavelength)

    def J_func(theta_arr: np.ndarray) -> np.ndarray:
        return numerical_jacobian(r_func, theta_arr)

    # 4. Optimization Loop
    solutions = []
    paths = []

    for i in range(num_solutions):
        print(f"--- Finding Solution {i+1} ---")
        eta_func, grad_eta_func = make_deflation_funcs(solutions)

        # We optimize a 1D array [theta]
        sol, path = good(
            r_func,
            J_func,
            grad_eta_func,
            initial_guess_angle,
            epsilon=0.1,    # Slightly larger shift for 1D landscape
            tol=1e-6,
            max_iter=50,
            verbose=False
        )

        # Normalize angle to [-pi, pi] for cleanliness
        normalized_sol = np.arctan2(np.sin(sol), np.cos(sol))

        solutions.append(normalized_sol)
        paths.append(path)

        cost = 0.5 * np.sum(r_func(normalized_sol)**2)
        print(f"Found Angle: {np.degrees(normalized_sol[0]):.1f}°, Cost: {cost:.2e}")

    if visualize:
        visualize_angle_landscape(
            r_func, solutions, paths, initial_guess_angle, true_angle
        )

    return solutions

def visualize_angle_landscape(r_func, solutions, paths, initial_guess, true_angle):
    """Visualizes the 1D cost function over angles 0 to 180 degrees."""

    # Generate 1D Cost Landscape
    # We scan from 0 to 180 degrees (typical forward facing array)
    angles_deg = np.linspace(0, 180, 1000)
    angles_rad = np.radians(angles_deg)
    costs = []

    for theta in angles_rad:
        r = r_func(np.array([theta]))
        costs.append(0.5 * np.sum(r**2))

    costs = np.array(costs)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot the cost curve
    ax.plot(angles_deg, costs, 'k-', linewidth=1.5, alpha=0.6, label='Cost Function')
    ax.fill_between(angles_deg, costs, alpha=0.1, color='blue')

    # 2. Plot Optimization Paths
    colors = ['cyan', 'magenta', 'orange', 'lime']
    for i, path in enumerate(paths):
        path_degs = np.degrees([p[0] for p in path])
        # Calculate cost for the path points
        path_costs = [0.5 * np.sum(r_func(p)**2) for p in path]

        ax.plot(path_degs, path_costs, 'o-',
                color=colors[i % len(colors)], markersize=4,
                label=f'Path {i+1}', alpha=0.8)

        # Mark the final found solution
        ax.plot(path_degs[-1], path_costs[-1], 'r*', markersize=12, zorder=10)

    # 3. Mark True Source
    true_deg = np.degrees(true_angle)
    true_cost = 0.5 * np.sum(r_func(np.array([true_angle]))**2)
    ax.axvline(true_deg, color='green', linestyle='--', alpha=0.5)
    ax.plot(true_deg, true_cost, 'g*', markersize=15, label='True Angle')

    # Formatting
    ax.set_title('1D Acoustic Localization: Angle Optimization', fontsize=14)
    ax.set_xlabel('Angle (Degrees)', fontsize=12)
    ax.set_ylabel('Cost (Least Squares)', fontsize=12)
    ax.set_yscale('log') # Log scale helps differentiate low minima
    ax.set_xlim(0, 180)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Optimize for angle starting from 45 degrees
    solve_acoustic_angle(
        num_solutions=4,
        initial_guess_angle=np.radians(45),
        visualize=True
    )
