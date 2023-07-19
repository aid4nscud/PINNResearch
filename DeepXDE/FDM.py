import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
alpha = 1.0  # Diffusivity
L = 1.0  # Length of domain
T = 1  # Time to solve until
Nx = 100  # Number of spatial points in grid
Ny = 100
Nt = 100  # Number of time steps

dx = L / (Nx - 1)  # Spatial step size
dy = L / (Ny - 1)
dt = min(dx**2/(4*alpha), dy**2/(4*alpha))  # Time step size

# Grids
x = np.linspace(0, L, Nx)  # x grid
y = np.linspace(0, L, Ny)  # y grid
t = np.linspace(0, T, Nt)  # time grid

# Initialize solution arrays
u = np.zeros((Nx, Ny, Nt))  

# Initial condition
u[:, :, 0] = np.zeros((Nx, Ny))

# Boundary conditions
u[:, 0, :] = 0  # left edge
u[:, -1, :] = 100  # right edge
u[0, :, :] = 0  # bottom edge
u[-1, :, :] = 0  # top edge

# Finite difference scheme
for k in range(0, Nt - 1):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u[i, j, k + 1] = (u[i, j, k] +
                              alpha * dt * ((u[i + 1, j, k] - 2 * u[i, j, k] + u[i - 1, j, k]) / dx ** 2 +
                                            (u[i, j + 1, k] - 2 * u[i, j, k] + u[i, j - 1, k]) / dy ** 2))

# Plot solution
fig = plt.figure(figsize=(7, 7))
im = plt.imshow(u[:, :, 0], extent=[0, L, 0, L], origin='lower', cmap='hot', interpolation="bilinear")
plt.colorbar(label="Temperature (K)")
plt.title('Heat equation solution')
plt.xlabel('x')
plt.ylabel('y')

# Adding text field for time
time_text = plt.text(0.1, 0.9, '', transform=plt.gca().transAxes)
# Define animation update function
def updatefig(k):
    im.set_array(u[:, :, k])
    current_time = k * T / Nt
    time_text.set_text('Time = %.2f' % current_time)
    return im, time_text,
# Create animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(Nt), interval=50, blit=True)

# Save as gif
ani.save('fdm_solution.gif', writer='pillow')


