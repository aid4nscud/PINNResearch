import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
alpha = 1.0  # Diffusivity
L = 1.0  # Length of domain
T = 1.0  # Time to solve until (in seconds)
Nx = 100  # Number of spatial points in x-direction
Ny = 100  # Number of spatial points in y-direction
Nt = 100  # Number of time steps

dx = L / (Nx - 1)  # Spatial step size
dy = L / (Ny - 1)
dt = min(dx**2/(4*alpha), dy**2/(4*alpha))  # Time step size

# Grids
x = np.linspace(0, L, Nx)  # x grid
y = np.linspace(0, L, Ny)  # y grid
t_data = np.linspace(0, T, Nt)  # time grid

# Initialize solution array
u = np.zeros((Nx, Ny, Nt))

# Initial condition
u[:, :, 0] = np.zeros((Nx, Ny))

# Boundary conditions
u[:, -1, :] = 100  # Right edge
u[:, 0, :] = 0  # Bottom edge
u[0, :, :] = 0  # Left edge
u[-1, :, :] = 0  # Top edge

# Finite difference scheme
for k in range(Nt - 1):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u[i, j, k + 1] = (u[i, j, k] +
                              alpha * dt * ((u[i + 1, j, k] - 2 * u[i, j, k] + u[i - 1, j, k]) / dx**2 +
                                            (u[i, j + 1, k] - 2 * u[i, j, k] + u[i, j - 1, k]) / dy**2))

# Function to animate and save the solution
def animate_solution(data, filename, title, label, t_data):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        data[:, :, 0],
        origin="lower",
        cmap="hot",
        interpolation="bilinear",
        extent=[0, L, 0, L],
    )
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def updatefig(k):
        im.set_array(data[:, :, k])
        ax.set_title(f"{title}, t = {t_data[k]:.2f} seconds")  # Update the title with current time step
        return [im]

    ani = animation.FuncAnimation(
        fig, updatefig, frames=range(data.shape[2]), interval=50, blit=True
    )
    ani.save(filename, writer="pillow")

# Call the function to animate and save the solution
animate_solution(u, "fdm_solution.gif", "Heat equation solution", "Temperature (K)", t_data)

# Show plot
plt.show()
