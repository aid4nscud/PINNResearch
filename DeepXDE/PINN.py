import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from FDM import HeatEquationFDM

# Configuration Parameters
ALPHA = 1.0
LENGTH = 1.0
WIDTH = 1.0
MAX_TIME = 1.0
LAYER_SIZE = [3] + [150] * 3 + [1]
ACTIVATION = "tanh"
INITIALIZER = "Glorot uniform"
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4
ITERATIONS = 10000
LOSS_WEIGHTS = [1, 20, 1, 1, 1, 10]

# FDM Parameters
NX = 100  # Number of spatial points in x-direction
NY = 100  # Number of spatial points in y-direction
NT = 100  # Number of time steps

# Create FDM solver
fdm_solver = HeatEquationFDM(ALPHA, LENGTH, MAX_TIME, NX, NY, NT)

# Solve FDM
print("Solving exactly using FDM\n")
fdm_solution = fdm_solver.solve()
print("Finished solving with FDM\n")


def main():
    # Check GPU Availability
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )

    # Define Domain
    geom = dde.geometry.Rectangle([0, 0], [LENGTH, WIDTH])
    timedomain = dde.geometry.TimeDomain(0, MAX_TIME)
    geotime = dde.geometry.GeometryXTime(geom, timedomain)

    # Define PDE
    def pde(X, u):
        du_X = tf.gradients(u, X)[0]
        du_x, du_y, du_t = du_X[:, 0:1], du_X[:, 1:2], du_X[:, 2:3]
        du_xx = tf.gradients(du_x, X)[0][:, 0:1]
        du_yy = tf.gradients(du_y, X)[0][:, 1:2]
        return du_t - ALPHA * (du_xx + du_yy)

    # Define Boundary Conditions
    def func_bc_right_edge(x):
        return np.where(np.isclose(x[:, 0], LENGTH), 100.0, 0.0)[:, None]

    def func_ic(x):
        return np.zeros((len(x), 1))

    def func_zero(x):
        return np.zeros_like(x)

    def solution(x):
        return fdm_solution[
            np.floor(x[:, 0] * (NX - 1)).astype(int),
            np.floor(x[:, 1] * (NY - 1)).astype(int),
            np.floor(x[:, 2] * (NT - 1)).astype(int),
        ][:, None]

    bc_right_edge = dde.DirichletBC(
        geotime,
        func_bc_right_edge,
        lambda x, on_boundary: on_boundary
        and np.isclose(x[0], LENGTH)
        and not np.isclose(x[1], 0)
        and not np.isclose(x[1], WIDTH),
    )
    bc_left = dde.NeumannBC(
        geotime,
        func_zero,
        lambda x, on_boundary: on_boundary
        and np.isclose(x[0], 0)
        and not np.isclose(x[1], 0)
        and not np.isclose(x[1], WIDTH),
    )
    bc_top = dde.NeumannBC(
        geotime,
        func_zero,
        lambda x, on_boundary: on_boundary
        and np.isclose(x[1], WIDTH)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], LENGTH),
    )
    bc_bottom = dde.NeumannBC(
        geotime,
        func_zero,
        lambda x, on_boundary: on_boundary
        and np.isclose(x[1], 0)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], LENGTH),
    )
    ic = dde.IC(geotime, func_ic, lambda _, on_initial: on_initial)

    # Define Training Data
    data = dde.data.TimePDE(
        geotime,
        pde,
        [bc_right_edge, bc_left, bc_top, bc_bottom, ic],
        num_domain=10000,
        num_boundary=500,
        num_initial=2000,
        solution=solution,
        num_test=10000,
        
    )

    pde_resampler = dde.callbacks.PDEPointResampler(period=10)

    # Define Neural Network Architecture and Model
    net = dde.nn.FNN(LAYER_SIZE, ACTIVATION, INITIALIZER)
    model = dde.Model(data, net)
    model = dde.Model(data, net)
    model.compile(OPTIMIZER, LEARNING_RATE, loss_weights=LOSS_WEIGHTS, metrics=["l2 relative error"])

    # Train Model
    # early_stopping = dde.callbacks.EarlyStopping(min_delta=5e-8, patience=1000)
    model.train(iterations=ITERATIONS, callbacks=[pde_resampler])

    # Generate Test Data
    x_data = np.linspace(0, LENGTH, num=100)
    y_data = np.linspace(0, WIDTH, num=100)
    t_data = np.linspace(0, 1, num=100)
    test_x, test_y, test_t = np.meshgrid(x_data, y_data, t_data)
    test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T

    # Predict Solution
    predicted_solution = model.predict(test_domain)
    residual = model.predict(test_domain, operator=pde)

    # Reshape Solution
    predicted_solution = predicted_solution.reshape(
        test_x.shape[0], test_y.shape[1], test_t.shape[2]
    )
    residual = residual.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])

    # Plot and Save Solution
    animate_solution(
        predicted_solution,
        "pinn_solution.gif",
        "Heat equation solution",
        "Temperature (K)",
        t_data,
    )
    animate_solution(residual, "pinn_residual.gif", "Residual plot", "Residual", t_data)


def animate_solution(data, filename, title, label, t_data):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        data[:, :, 0],
        origin="lower",
        cmap="hot",
        interpolation="bilinear",
        extent=[0, LENGTH, 0, LENGTH],
    )
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def updatefig(k):
        im.set_array(data[:, :, k])
        ax.set_title(
            f"{title}, t = {t_data[k]:.2f}"
        )  # Update the title with the current time step
        return [im]

    ani = animation.FuncAnimation(
        fig, updatefig, frames=range(data.shape[2]), interval=50, blit=True
    )
    ani.save(filename, writer="pillow")


if __name__ == "__main__":
    main()
