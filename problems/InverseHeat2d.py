import deepxde as dde
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Initialize the value of alpha
ALPHA = dde.Variable(1e-3)
WIDTH = 1
LENGTH = 1
T_END = 1
BATCH_SIZE = 256
ITERATIONS = 10000
LOSS_WEIGHTS = [
    10,
    1,
    1,
    1,
    1,
    1,
    100,
]  # Weights for different components of the loss function
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4

# Define the PDE
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

# Load the data
data_dict = np.load("temperature_data.npz")
X_data = data_dict["x_data"]
Y_data = data_dict["y_data"]
T_data = data_dict["T_data"]


# Flatten and stack to create observation points
observe_x = np.hstack(
    (
        X_data.flatten()[:, None],
        Y_data.flatten()[:, None],
        np.ones_like(X_data.flatten())[:, None],
    )
)
observe_y = T_data.flatten()[:, None]

x_min = observe_x[:, 0].min()
x_max = observe_x[:, 0].max()
observe_x[:, 0] = (observe_x[:, 0] - x_min) / (x_max - x_min)

y_min = observe_x[:, 1].min()
y_max = observe_x[:, 1].max()
observe_x[:, 1] = (observe_x[:, 1] - y_min) / (y_max - y_min)

# Normalize observe_y
T_min = observe_y.min()
T_max = observe_y.max()
observe_y = (observe_y - T_min) / (T_max - T_min)


# Define observation points
observed_data = dde.PointSetBC(observe_x, observe_y)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_l, bc_r, bc_up, bc_low, ic, observed_data],
    num_domain=2000,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
    num_test=50000,
)

# Network architecture
layer_size = [3] + [60] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(lambda _, y: abs(y))
model = dde.Model(data, net)

# Compile model
model.compile(
    OPTIMIZER,
    lr=LEARNING_RATE,
    loss_weights=LOSS_WEIGHTS,
    external_trainable_variables=[ALPHA],
)

# Define callback to save ALPHA
variable = dde.callbacks.VariableValue(ALPHA, period=1000)

# Train the model
losshistory, train_state = model.train(
    iterations=ITERATIONS, batch_size=BATCH_SIZE, callbacks=[variable]
)

# Residual Adaptive Refinement (RAR)
X = geomtime.random_points(1000)
err = 1
while err > 0.01:
    f = model.predict(X, operator=pde)
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))
    x_id = np.argmax(err_eq)
    print("Adding new point:", X[x_id], "\n")
    data.add_anchors(X[x_id])
    # Stop training if the model isn't learning anymore
    early_stopping = dde.callbacks.EarlyStopping(min_delta=LEARNING_RATE, patience=2000)
    model.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
    model.train(
        iterations=100, disregard_previous_best=True, batch_size=BATCH_SIZE, callbacks=[early_stopping]
    )
    model.compile("L-BFGS-B")
    dde.optimizers.set_LBFGS_options(
        maxcor=100,
    )
    losshistory, train_state = model.train(batch_size=BATCH_SIZE, callbacks=[variable])

ALPHA = tf.math.abs(ALPHA)
ALPHA_float = float(ALPHA.numpy())  # Convert the tensor to float
print("PINN Prediction of Alpha Parameter " + str(ALPHA_float) + "\n")

# Save and plot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
plt.show()
plt.savefig("loss_history_plot_inverseHeat2d")
plt.close()
