import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
W = 1
H = 1
Nx = 101
Ny = 101
ALPHA = 0.01
total_time = 1
dt = 0.001
output_times = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Discretization
dx = W / (Nx - 1)
dy = H / (Ny - 1)

# Mesh grid
x = np.linspace(0, W, Nx)
y = np.linspace(0, H, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.zeros((Nx, Ny))

# Time-stepping loop
time = 0
for output_time in output_times:
    while time < output_time:
        T_new = T.copy()
        
        # Internal nodes
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                T_new[i, j] = T[i, j] + ALPHA * dt * (
                    (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dx**2
                    + (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dy**2
                )

        # Neumann boundary conditions (zero-flux)
        T_new[0, :] = T_new[1, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]

        # Dirichlet boundary condition
        T_new[-1, :] = 100

        T = T_new
        time += dt

    # Visualization
    plt.contourf(X, Y, T.T, cmap="jet", levels=100, vmin=0, vmax=100)
    plt.colorbar()
    plt.title(f"Temperature Distribution at t={output_time}")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    plt.show()
