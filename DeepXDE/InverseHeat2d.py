import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import re

# true values
T_START = 0
T_END = WIDTH = LENGTH = ALPHA = 1.0


# Load training data
def load_training_data(filename, num):
    data_list = np.load(filename, allow_pickle=True)["data_list"]

    x = []
    y = []
    t = []
    T = []

    for data_dict in data_list:
        x.extend(data_dict["x_data"][:num])
        y.extend(data_dict["y_data"][:num])
        t.extend(data_dict["t_data"][:num])
        T.extend(data_dict["T_data"][:num])

    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    t = np.array(t).reshape(-1, 1)
    T = np.array(T).reshape(-1, 1)  # Temperature data

    return x, y, t, T


# Parameters to be identified
ALPHA = dde.Variable(0.0)


# Define PDE
def pde(X, T):
    # Calculate second derivatives (Hessians) of T with respect to X in both dimensions
    dT_xx = dde.grad.hessian(T, X, i=0, j=0)
    dT_yy = dde.grad.hessian(T, X, i=1, j=1)

    # Calculate first derivative (Jacobian) of T with respect to X in time dimension
    dT_t = dde.grad.jacobian(T, X, j=2)

    # Return the defined PDE
    return dT_t - (ALPHA * dT_xx + dT_yy)


# Define boundary conditions
def boundary_right(X, on_boundary):
    x, _, _ = X
    return on_boundary and np.isclose(x, WIDTH)  # Check if on the right boundary


def boundary_left(X, on_boundary):
    x, _, _ = X
    return on_boundary and np.isclose(x, 0)  # Check if on the left boundary


def boundary_top(X, on_boundary):
    _, y, _ = X
    return on_boundary and np.isclose(y, LENGTH)  # Check if on the upper boundary


def boundary_bottom(X, on_boundary):
    _, y, _ = X
    return on_boundary and np.isclose(y, 0)  # Check if on the lower boundary


# Define initial condition
def boundary_initial(X, on_initial):
    _, _, t = X
    return on_initial and np.isclose(t, 0)  # Check if at the initial time


# Initialize a function for the temperature field
def init_func(X):
    t = np.zeros((len(X), 1))  # Temperature is zero everywhere at the T_START
    return t


# Define Dirichlet and Neumann boundary conditions
def constraint_right(X):
    return np.ones((len(X), 1))  # On the right boundary, temperature is kept at 1


def func_zero(X):
    return np.zeros(
        (len(X), 1)
    )  # On the other boundaries, the derivative of temperature is kept at 0 (Neumann condition)


# Define geometry and time domains
geom = dde.geometry.Rectangle([0, 0], [WIDTH, LENGTH])  # Geometry domain
timedomain = dde.geometry.TimeDomain(0, T_END)  # Time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain)  # Space-time domain

# Define boundary conditions and initial condition
bc_l = dde.NeumannBC(geomtime, func_zero, boundary_left)  # Left boundary
bc_r = dde.DirichletBC(geomtime, constraint_right, boundary_right)  # Right boundary
bc_up = dde.NeumannBC(geomtime, func_zero, boundary_top)  # Upper boundary
bc_low = dde.NeumannBC(geomtime, func_zero, boundary_bottom)  # Lower boundary
ic = dde.IC(geomtime, init_func, boundary_initial)  # Initial condition

# Get the training data: num = 5000
ob_x, ob_y, ob_t, ob_T = load_training_data("temperature_data.npz", num=5000)
ob_xyt = np.hstack((ob_x, ob_y, ob_t))
observe_T = dde.icbc.PointSetBC(ob_xyt, ob_T, component=0)
# Training datasets and Loss
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_l, bc_r, bc_up, bc_low, observe_T],
    num_domain=1000,
    num_boundary=400,
    num_initial=200,
    anchors=ob_xyt,
)

# Neural Network setup
layer_size = [3] + [50] * 6 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

# callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([ALPHA], period=100, filename=fnamevar)

# Compile, train and save model
model.compile("adam", lr=1e-3, external_trainable_variables=ALPHA)
loss_history, train_state = model.train(
    iterations=10000,
    callbacks=[variable],
    display_every=1000,
    disregard_previous_best=True,
)

model.compile("adam", lr=1e-4, external_trainable_variables=[ALPHA])
loss_history, train_state = model.train(
    iterations=10000,
    callbacks=[variable],
    display_every=1000,
    disregard_previous_best=True,
)
dde.saveplot(loss_history, train_state, issave=True, isplot=True)
plt.show()
plt.savefig("loss_history_plot_InverseHeat2d")
