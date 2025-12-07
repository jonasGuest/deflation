import numpy as np
import matplotlib.pyplot as plt


# --- 1. Simulation Setup (The "Real World") ---
def generate_perfect_data():
    # Speed of sound
    c = 343.0

    # Frequency: 2000 Hz (Wavelength = ~17cm)
    f = 2000.0
    wavelength = c / f

    # Microphone Array: 5 mics in a line, spaced 20cm apart
    # NOTE: Spacing (0.2m) > Wavelength (0.17m). This creates SPATIAL ALIASING (Multiple Minima)!
    mics = np.array([
        [0.0, 0.0],
        [0.2, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [0.8, 0.0]
    ])

    # True Source Location (The "Answer")
    # Placed at x=0.4 (center of array), y=1.0 (1 meter away)
    true_source = np.array([0.4, 1.0])

    # Calculate Phase Shifts relative to Mic 0
    dists = np.linalg.norm(mics - true_source, axis=1)
    dist_diffs = dists - dists[0]

    # Theoretical Phase = (2 * pi * dist_diff) / wavelength
    # We add modulo 2pi because in the real world, we only measure 0 to 2pi
    measured_phases = (2 * np.pi * dist_diffs / wavelength) % (2 * np.pi)

    return mics, measured_phases, wavelength, true_source


# --- 2. The Residual Function for Solver ---
def phase_residual(source_pos, mics, measured_phases, wavelength):
    """
    Computes the residual vector for a single frequency sine wave.
    Input:
        source_pos: [x, y] or [x, y] optimization variable
        mics: [N, 2] known positions
        measured_phases: [N] known phases (radians)
        wavelength: float
    Returns:
        residual: [2*N] vector (Cos errors and Sin errors)
    """
    # Ensure source_pos is 3D (if solver passes 2D)

    # 1. Calculate Model Distance Differences (w.r.t Mic 0)
    dists = np.linalg.norm(mics - source_pos, axis=1)
    dist_diffs = dists - dists[0]

    # 2. Calculate Model Phase (Unwrapped)
    model_phases = 2 * np.pi * dist_diffs / wavelength

    phase_diffs = measured_phases - measured_phases[0]
    # 3. Calculate Residuals on the Unit Circle
    # We compare the point (cos_meas, sin_meas) vs (cos_model, sin_model)

    # Real part error (Cos difference)
    r_cos = np.cos(phase_diffs) - np.cos(model_phases)

    # Imag part error (Sin difference)
    r_sin = np.sin(phase_diffs) - np.sin(model_phases)

    # Concatenate to make a vector of size 2N
    return np.concatenate([r_cos, r_sin])


# --- 3. Visualization of the Cost Surface ---
def visualize_landscape():
    mics, measured_phases, wavelength, true_source = generate_perfect_data()

    # Grid search to visualize the "Least Squares" cost surface
    x_range = np.linspace(-1, 2, 500)
    y_range = np.linspace(-0.1, 2, 500)  # Keep y positive (in front of array)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    print(f"Wavelength: {wavelength:.3f}m")
    print(f"Mic Spacing: {np.linalg.norm(mics[1] - mics[0]):.3f}m")
    print("Generating cost surface...")

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            guess = np.array([X[i, j], Y[i, j]])
            r = phase_residual(guess, mics, measured_phases, wavelength)
            # Cost = 0.5 * sum(r^2)
            Z[i, j] = 0.5 * np.dot(r,r)

    plt.figure(figsize=(10, 8))
    # Log scale to see the minima better
    plt.contourf(X, Y, np.log1p(Z), levels=50, cmap='viridis')
    plt.colorbar(label='Log Cost Function')

    # Plot Mics
    plt.plot(mics[:, 0], mics[:, 1], 'k^', markersize=10, label='Microphones')

    # Plot True Source
    plt.plot(true_source[0], true_source[1], 'r*', markersize=15, label='True Source')

    plt.title("Acoustic Phase Optimization Landscape\n(Notice the Multiple Minima / Grating Lobes!)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    print("Close the plot to finish.")
    plt.show()


if __name__ == "__main__":
    visualize_landscape()