# Import necessary libraries
import deepxde as dde  # Deep learning framework for solving differential equations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations in Python
import numpy as np  # For numerical operations
from deepxde.backend import tf  # Tensorflow backend for DeepXDE
import matplotlib.animation as animation  # For creating animations
from matplotlib.animation import (
    FuncAnimation,
)  # Function-based interface to create animations

# Constants/Network Parameters
T_START = 0
T_END = WIDTH = LENGTH = 1.0
EPSILON = 0.0125
SAMPLE_POINTS = 2000
ARCHITECTURE = (
    [3] + [60] * 5 + [1]
)  # Network architecture ([input_dim, hidden_layer_1_dim, ..., output_dim])
ACTIVATION = "tanh"  # Activation function
INITIALIZER = "Glorot uniform"  # Weights initializer
LEARNING_RATE = 1e-3  # Learning rate
LOSS_WEIGHTS = [1, 1, 100]  # Weights for different components of the loss function
ITERATIONS = 10000  # Number of training iterations
OPTIMIZER = "adam"  # Optimizer for the first part of the training
BATCH_SIZE = 32  # Batch size


# Define Allen-Cahn PDE
def pde(X, u):
    du_t = dde.grad.jacobian(u, X, j=2)
    du_xx = dde.grad.hessian(u, X, i=0, j=0)
    du_yy = dde.grad.hessian(u, X, i=1, j=1)
    return du_t - EPSILON**2 * (du_xx + du_yy) + 10 * (u - u**3)


# Define boundary conditions
def boundary_condition(X, on_boundary):
    return on_boundary


# Define initial condition
def boundary_initial(X, on_initial):
    _, _, t = X
    return on_initial and np.isclose(t, 0)  # Check if at the initial time


# Initialize a function for the temperature field
def init_func(X):
    t = np.random.uniform(
        -1, 1, (len(X), 1)
    )  # Temperature is randomly distributed between -0.05 and 0.05 everywhere at the start
    return t


# Define Neumann boundary condition
def func_zero(X):
    return np.zeros(
        (len(X), 1)
    )  # On the other boundaries, the derivative of temperature is kept at 0 (Neumann condition)


# Define geometry and time domains
geom = dde.geometry.Rectangle([0, 0], [WIDTH, LENGTH])  # Geometry domain
timedomain = dde.geometry.TimeDomain(0, T_END)  # Time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain)  # Space-time domain

# Define boundary conditions and initial condition
bc = dde.NeumannBC(geomtime, func_zero, boundary_condition)  # Left boundary
ic = dde.IC(geomtime, init_func, boundary_initial)  # Initial condition

# Define data for the PDE
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=SAMPLE_POINTS,
    num_boundary=int(SAMPLE_POINTS / 8),
    num_initial=int(SAMPLE_POINTS),
)

# Define the neural network model
net = dde.maps.FNN(ARCHITECTURE, ACTIVATION, INITIALIZER)  # Feed-forward neural network
model = dde.Model(data, net)  # Create the model

# Compile the model with the chosen optimizer, learning rate and loss weights
model.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
# Train the model
losshistory, trainstate = model.train(
    iterations=ITERATIONS,
    batch_size=BATCH_SIZE,
)

# Residual Adaptive Refinement (RAR)
X = geomtime.random_points(1000)
f = model.predict(X, operator=pde)
err_eq = np.absolute(f)
err = np.mean(err_eq)
print("Mean residual: %.3e" % (err))
x_id = np.argmax(err_eq)
print("Adding new point:", X[x_id], "\n")
data.add_anchors(X[x_id])
# Stop training if the model isn't learning anymore
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
model.compile(OPTIMIZER, lr=LEARNING_RATE)
model.train(
    iterations=100,
    disregard_previous_best=True,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
)
model.compile("L-BFGS-B")
dde.optimizers.set_LBFGS_options(
    maxcor=100,
)
losshistory, train_state = model.train(
    batch_size=BATCH_SIZE,
)

dde.saveplot(losshistory, trainstate, issave=True, isplot=True)
plt.show()
plt.savefig("loss_history_plot_AllenCahn")
plt.close()

# Predict the solution at different time points and create an animation
fig, ax = plt.subplots()
ax = fig.add_subplot(111)

# Set up the grid
nelx = 100  # Number of elements in x direction
nely = 100  # Number of elements in y direction
timesteps = 101  # Number of time steps
x = np.linspace(0, 1, nelx + 1)  # x coordinates
y = np.linspace(0, 1, nely + 1)  # y coordinates
t = np.linspace(0, 1, timesteps)  # Time points

# Prepare the data for the prediction
test_x, test_y, test_t = np.meshgrid(x, y, t)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T

# Predict Solution and Residual
predicted_solution = model.predict(test_domain)
predicted_solution = predicted_solution.reshape(test_x.shape)
residual = model.predict(test_domain, operator=pde)
residual = residual.reshape(test_x.shape)  # Reshape residuals


# Animation function
def animate_solution(data, filename, title, label, t_data):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Create initial image and colorbar
    im = ax.imshow(
        data[:, :, 0],
        origin="lower",
        cmap="jet",
        interpolation="bilinear",
        extent=[0, 1, 0, 1],
    )
    cb = plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Update function for the frames
    def updatefig(k):
        # Update image data
        im.set_array(data[:, :, k])
        im.set_clim(
            vmin=data[:, :, k].min(), vmax=data[:, :, k].max()
        )  # Update the color limits

        # Update colorbar
        cb.update_normal(im)

        # Update title
        ax.set_title(f"{title}, t = {t_data[k]:.2f}")

        return [im]

    ani = animation.FuncAnimation(
        fig, updatefig, frames=range(data.shape[2]), interval=50, blit=True
    )
    ani.save(filename, writer="ffmpeg")


# Create and save the solution animation
animate_solution(
    predicted_solution,
    "pinn_AC_solution.mp4",
    "Surface: Dependent variable T (u)",
    " ",
    t,
)

# Create and save the residuals animation
animate_solution(
    residual,
    "pinn_AC_residual.mp4",
    "Residual",
    "Residual",
    t,
)
