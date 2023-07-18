import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

# Configuration Parameters
ALPHA = 1.0
LENGTH = 1.0
WIDTH = 1.0
MAX_TIME = 1.0
LAYER_SIZE = [3] + [64] * 4 + [1]
ACTIVATION = "tanh"
INITIALIZER = "Glorot uniform"
OPTIMIZER = "L-BFGS"
LEARNING_RATE = 1e-4
ITERATIONS = 10000

# L-BFGS config
dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=1000, maxfun=None)


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

    bc_right_edge = dde.DirichletBC(
        geotime,
        func_bc_right_edge,
        lambda _, on_boundary: on_boundary,
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
        num_domain=8000,
        num_boundary=3000,
        num_initial=2000,
        num_test=1000,
    )

    pde_resampler = dde.callbacks.PDEPointResampler(period=50)

    # Define Neural Network Architecture and Model
    net = dde.nn.FNN(LAYER_SIZE, ACTIVATION, INITIALIZER)
    model = dde.Model(data, net)
    model.compile(OPTIMIZER, LEARNING_RATE)

    # Train Model
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
        )  # Update the title with current time step
        return [im]

    ani = animation.FuncAnimation(
        fig, updatefig, frames=range(data.shape[2]), interval=50, blit=True
    )
    ani.save(filename, writer="pillow")


if __name__ == "__main__":
    main()
