import good
import numpy as np
import matplotlib.pyplot as plt

def scalar_to_residual(f):
    """Convert scalar function to residual format."""
    def residual(x):
        return np.array([f(x)])
    return residual

def scalar_to_jacobian(f):
    """Convert scalar function to Jacobian format."""
    def jacobian(x):
        grad = good.numeric_gradient(f, x)
        return grad.reshape(1, -1)  # 1×n matrix
    return jacobian


def jacobian_ackley(xy):
    from functions import ackley
    v = ackley(xy)
    return good.numeric_gradient(ackley, xy)

def test_ackley_func():
    from functions import ackley
    import numpy as np

    _, grad_eta = good.make_deflation_funcs([])

    x0 = np.array([3.0, 10.3])
    (minima, history) = good.good_deflated_gauss_newton(
        scalar_to_residual(ackley),
        scalar_to_jacobian(ackley),
        grad_eta,
        x0,
        max_iter=100
    )

    print("Found minimum at:", minima)
    print("Function value at minimum:", ackley(minima))

    # Visualization
    # Create grid for plotting
    x_range = np.linspace(-5, 5, 300)
    y_range = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x_range, y_range)

    # Compute Ackley function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ackley(np.array([X[i, j], Y[i, j]]))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot function as contour
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label='Ackley Function Value')

    # Plot optimization path
    history_array = np.array(history)
    ax.plot(history_array[:, 0], history_array[:, 1], 'ro-',
            label='Optimization Path', markersize=4, linewidth=2, alpha=0.9)

    # Plot start and end points
    ax.plot(x0[0], x0[1], 'g*', markersize=20,
            markeredgewidth=2, markeredgecolor='white', label='Start', zorder=10)
    ax.plot(minima[0], minima[1], 'r*', markersize=20,
            markeredgewidth=2, markeredgecolor='white', label='Minimum Found', zorder=10)

    # Plot true global minimum at (0, 0)
    ax.plot(0, 0, 'w*', markersize=20,
            markeredgewidth=2, markeredgecolor='black', label='Global Minimum', zorder=10)

    # Add iteration numbers at key points
    step = max(1, len(history) // 10)
    for i in range(0, len(history), step):
        ax.annotate(f'{i}', history_array[i],
                   textcoords="offset points", xytext=(5,5),
                   fontsize=8, color='white', fontweight='bold')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Ackley Function Optimization\nIterations: {len(history)-1}, Final Value: {ackley(minima):.6f}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('ackley_optimization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to ackley_optimization.png")
    plt.show()


if __name__ == "__main__":
    test_ackley_func()