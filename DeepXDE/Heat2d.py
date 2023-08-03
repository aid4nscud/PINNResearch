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
T_END = WIDTH = LENGTH = ALPHA = 1.0
NUM_DOMAIN = 30000  # Number of training samples in the domain
NUM_BOUNDARY = 8000  # Number of training samples on the boundary
NUM_INITIAL = 20000  # Number of training samples for initial conditions
ARCHITECTURE = (
    [3] + [60] * 5 + [1]
)  # Network architecture ([input_dim, hidden_layer_1_dim, ..., output_dim])
ACTIVATION = "tanh"  # Activation function
INITIALIZER = "Glorot uniform"  # Weights initializer
LEARNING_RATE = 1e-3  # Learning rate
LOSS_WEIGHTS = [
    10,
    1,
    1,
    1,
    1,
    10,
]  # Weights for different components of the loss function
ITERATIONS = 10000  # Number of training iterations
OPTIMIZER = "adam"  # Optimizer for the first part of the training
BATCH_SIZE = 256  # Batch size


# Define PDE
def pde(X, T):
    # Calculate second derivatives (Hessians) of T with respect to X in both dimensions
    dT_xx = dde.grad.hessian(T, X, j=0)
    dT_yy = dde.grad.hessian(T, X, j=1)

    # Calculate first derivative (Jacobian) of T with respect to X in time dimension
    dT_t = dde.grad.jacobian(T, X, j=2)

    # Return the defined PDE
    return dT_t - (ALPHA * (dT_xx + dT_yy))


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

# Define data for the PDE
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_l, bc_r, bc_up, bc_low, ic],
    num_domain=NUM_DOMAIN,
    num_boundary=NUM_BOUNDARY,
    num_initial=NUM_INITIAL,
)

# Define the neural network model
net = dde.maps.FNN(ARCHITECTURE, ACTIVATION, INITIALIZER)  # Feed-forward neural network
net.apply_output_transform(lambda _, y: abs(y))
model = dde.Model(data, net)  # Create the model

# Compile the model with the chosen optimizer, learning rate and loss weights
model.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
# Train the model
losshistory, trainstate = model.train(
    iterations=ITERATIONS,
    batch_size=BATCH_SIZE,
)
# Re-compile the model with the L-BFGS optimizer
model.compile("L-BFGS-B")
dde.optimizers.set_LBFGS_options(
    maxcor=50,
)
# Train the model again with the new optimizer
losshistory, train_state = model.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)
dde.saveplot(losshistory, trainstate, issave=True, isplot=True)

# Predict the solution at different time points and create an animation
fig, ax = plt.subplots()
ax = fig.add_subplot(111)
nelx = 100  # Number of elements in x direction
nely = 100  # Number of elements in y direction
timesteps = 101  # Number of time steps
x = np.linspace(0, 1, nelx + 1)  # x coordinates
y = np.linspace(0, 1, nely + 1)  # y coordinates
t = np.linspace(0, 1, timesteps)  # Time points
delta_t = t[1] - t[0]  # Time step
xx, yy = np.meshgrid(x, y)

# Prepare the data for the prediction
x_ = np.zeros(shape=((nelx + 1) * (nely + 1),))
y_ = np.zeros(shape=((nelx + 1) * (nely + 1),))
for c1, ycor in enumerate(y):
    for c2, xcor in enumerate(x):
        x_[c1 * (nelx + 1) + c2] = xcor
        y_[c1 * (nelx + 1) + c2] = ycor

# Predict the solution and residual at each time point
Ts = []  # List to store the solution at each time point
residuals = []  # List to store the residuals at each time point

for time in t:
    t_ = np.ones(
        (nelx + 1) * (nely + 1),
    ) * (time)
    X = np.column_stack((x_, y_))  # Making 2d array with x and y
    X = np.column_stack((X, t_))  # Making 3d array with the 2d array and t

    T = model.predict(X)  # Predict the solution
    T = (
        T * 100
    )  # Apply scaling to the prediction of the network, since we scaled down our BC by 100
    T = T.reshape(
        T.shape[0],
    )
    T = T.reshape(nelx + 1, nely + 1)
    Ts.append(T)

    residual = T.reshape(T.shape[1])  # Predict the residuals
    residual = residual.reshape(nelx + 1, nely + 1)
    residuals.append(residual)


# Function to plot the heatmap of the residuals
def plot_residual_map(residual, time):
    plt.clf()  # Clear the current plot figure
    plt.title(f"Time = {round(time*delta_t, ndigits=2)}     Residual")
    plt.xlabel("x")  # x label
    plt.ylabel("y")  # y label
    plt.pcolor(xx, yy, residual, cmap="jet")  # Plot the residuals as a colored heatmap
    plt.colorbar()  # Add a colorbar to the plot
    return plt


# Function to update the plot for each frame of the animation
def animate_residual(k):
    plot_residual_map(residuals[k], k)


# Function to plot the heatmap of the solution
def plotheatmap(T, time):
    plt.clf()  # Clear the current plot figure
    plt.title(
        f"Time = {round(time*delta_t, ndigits=2)}     Surface: Dependent variable T (K)"
    )
    plt.xlabel("x")  # x label
    plt.ylabel("y")  # y label
    plt.pcolor(xx, yy, T, cmap="jet")  # Plot the solution as a colored heatmap
    plt.colorbar()  # Add a colorbar to the plot
    return plt


# Function to update the plot for each frame of the animation
def animate(k):
    plotheatmap(Ts[k], k)


# Create the animation
anim = animation.FuncAnimation(
    plt.figure(), animate, interval=100, frames=len(t), repeat=False
)

# Create the animation
anim_residual = animation.FuncAnimation(
    plt.figure(), animate_residual, interval=100, frames=len(t), repeat=False
)

# Save the animation as a mp4 file
anim_residual.save("pinn_heat2d_residual.mp4", writer="ffmpeg")

# Save the animation as a mp4 file
anim.save("pinn_heat2d_solution.mp4", writer="ffmpeg")
