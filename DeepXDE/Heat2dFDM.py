import numpy as np
import matplotlib.pyplot as plt
import scipy.io

data_list = []
num_runs = 5000

for run in range(1, num_runs):
    print(str.format("Starting run {}", run))
    # Domain parameters
    W = 1
    H = 1
    Nx = 101
    Ny = 101
    ALPHA = 1 / num_runs  # You can now change ALPHA dynamically

    # Discretization
    dx = W / (Nx - 1)
    dy = H / (Ny - 1)
    dt = min(dx**2, dy**2) / (4 * ALPHA)  # Dynamically computed time step
    total_time = 1
    output_times = np.linspace(
        0, total_time, 11
    )  # You can modify the number of output times as needed

    # Mesh grid
    x = np.linspace(0, W, Nx)
    y = np.linspace(0, H, Ny)
    X, Y = np.meshgrid(x, y)

    # Initial condition
    T = np.zeros((Nx, Ny))

    # Time-stepping loop

    x_data = []
    y_data = []
    t_data = []
    T_data = []

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

            # Append the data for the current output time
            for i in range(Nx):
                for j in range(Ny):
                    x_data.append(x[i])
                    y_data.append(y[j])
                    t_data.append(output_time)
                    T_data.append(T[i, j])

        # Visualization
        # plt.contourf(X, Y, T, cmap="jet", levels=100, vmin=T.min(), vmax=T.max())
        # plt.colorbar()
        # plt.title(f"Temperature Distribution at t={output_time}")
        # plt.xlabel("x-direction")
        # plt.ylabel("y-direction")
        # plt.show()

    # Append the data
    for i in range(Nx):
        for j in range(Ny):
            x_data.append(x[i])
            y_data.append(y[j])
            t_data.append(output_time)
            T_data.append(T[i, j])

    # Convert lists to arrays
    data_dict = {
        "x_data": np.array(x_data),
        "y_data": np.array(y_data),
        "t_data": np.array(t_data),
        "T_data": np.array(T_data),
    }
    data_list.append(data_dict)

    # Save to .mat file
scipy.io.savemat("temperature_data.mat", {"data": data_list})
