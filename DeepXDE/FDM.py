import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class HeatEquationFDM:
    def __init__(self, alpha, L, T, Nx, Ny, Nt):
        """
        Initializes the finite difference method solver with the given parameters.

        Parameters:
        alpha (float): Diffusivity
        L (float): Length of the domain
        T (float): Time to solve until (in seconds)
        Nx (int): Number of spatial points in x-direction
        Ny (int): Number of spatial points in y-direction
        Nt (int): Number of time steps
        """
        self.alpha = alpha
        self.L = L
        self.T = T
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.dx = L / (Nx - 1)
        self.dy = L / (Ny - 1)
        self.dt = min(self.dx**2 / (4 * alpha), self.dy**2 / (4 * alpha))
        self.u = np.zeros((Nx, Ny, Nt))

    def solve(self):
        """
        Solves the heat equation using finite difference method.

        Returns:
        ndarray: Solved temperature distribution in 3D (x, y, time).
        """
        # Initial condition
        self.u[:, :, 0] = np.zeros((self.Nx, self.Ny))

        # Boundary conditions
        self.u[:, -1, :] = 100  # Right edge
        self.u[:, 0, :] = 0  # Bottom edge
        self.u[0, :, :] = 0  # Left edge
        self.u[-1, :, :] = 0  # Top edge

        # Finite difference scheme
        for k in range(self.Nt - 1):
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    self.u[i, j, k + 1] = self.u[i, j, k] + self.alpha * self.dt * (
                        (
                            self.u[i + 1, j, k]
                            - 2 * self.u[i, j, k]
                            + self.u[i - 1, j, k]
                        )
                        / self.dx**2
                        + (
                            self.u[i, j + 1, k]
                            - 2 * self.u[i, j, k]
                            + self.u[i, j - 1, k]
                        )
                        / self.dy**2
                    )
        return self.u

    def get_xyt_grids(self):
        """
        Generates the spatial and temporal grids.

        Returns:
        tuple: x, y, and t grids as numpy arrays.
        """
        x_data = np.linspace(0, self.L, self.Nx)
        y_data = np.linspace(0, self.L, self.Ny)
        t_data = np.linspace(0, self.T, self.Nt)
        return x_data, y_data, t_data

    def animate_solution(self):
        """
        Creates an animated heat map of the solution.
        """
        fig, ax = plt.subplots()  # Changed this line to get the ax object
        im = ax.imshow(self.u[:, :, 0], animated=True, origin='lower', extent=[0, self.L, 0, self.L], cmap='hot')
        txt = ax.text(0.8, 0.1, '', transform=ax.transAxes)  # Text placeholder

        x, y, t = self.get_xyt_grids()

        def updatefig(i):
            im.set_array(self.u[:, :, i])
            txt.set_text(f"t = {t[i]:.2f}")  # Set the text to current time
            return im, txt  # Return the updated heatmap and text

        ani = animation.FuncAnimation(
            fig, updatefig, frames=self.Nt, interval=50, blit=True
        )
        ani.save("FDM_Solution.gif")

if __name__ == "__main__":
    heat_solver = HeatEquationFDM(alpha=1.0, L=1.0, T=1.0, Nx=100, Ny=100, Nt=101)
    heat_solver.solve()
    heat_solver.animate_solution()
