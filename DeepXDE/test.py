import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Configuration Parameters
DIFF_COEF = 1.0
PLATE_LENGTH = 1.0
T_FINAL = 1.0
HOT_EDGE_TEMP = 100.0
COLD_EDGE_TEMP = 0
INITIAL_TEMP = 0
NUM_TIMEFRAMES = 100000
NUM_X_POINTS = 100
NUM_Y_POINTS = 100
NUM_PLOTS = 100

# Domain definition
geom = dde.geometry.Rectangle([0, 0], [PLATE_LENGTH, PLATE_LENGTH])
timedomain = dde.geometry.TimeDomain(0, T_FINAL)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# PDE definition
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return dy_t - DIFF_COEF * (dy_xx + dy_xx.T)


# Boundary conditions
def bc(x, on_boundary):
    return on_boundary


# Initial condition
def ic(x):
    return INITIAL_TEMP * np.ones_like(x)


# Define the problem
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc],
    num_domain=NUM_X_POINTS * NUM_Y_POINTS,
    num_boundary=4 * NUM_X_POINTS + 4 * NUM_Y_POINTS,
    num_initial=NUM_X_POINTS * NUM_Y_POINTS,
    func_ic=ic,
    num_test=NUM_TIMEFRAMES,
)

# Define the neural network architecture
layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

# Define the model
model = dde.Model(data, net)

# Train the model
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=NUM_TIMEFRAMES)

# Generate test points
x_test, _ = data.pde.get_test_points()

# Predict the solution
y_test = model.predict(x_test)

# Reshape the solution
u_test = y_test.reshape((NUM_TIMEFRAMES, NUM_X_POINTS, NUM_Y_POINTS))

# Plot and Save Solution


def animate_solution(data, filename, title, label):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(data[0, :, :], origin="lower", cmap="hot", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def updatefig(k):
        im.set_array(data[k, :, :])
        ax.set_title(
            f"{title}, t = {k / NUM_TIMEFRAMES * T_FINAL:.2f}"
        )  # Update the title with current time step
        return [im]

    ani = animation.FuncAnimation(
        fig, updatefig, frames=range(NUM_TIMEFRAMES), interval=50, blit=True
    )
    ani.save(filename, writer="pillow")


animate_solution(u_test, "test.gif", "Heat equation solution", "Temperature")
