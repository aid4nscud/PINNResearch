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
SAMPLE_POINTS = 5000
ARCHITECTURE = (
    [3] + [60] * 5 + [1]
)  # Network architecture ([input_dim, hidden_layer_1_dim, ..., output_dim])
ACTIVATION = "tanh"  # Activation function
INITIALIZER = "Glorot uniform"  # Weights initializer
LEARNING_RATE = 1e-3  # Learning rate
LOSS_WEIGHTS = [1, 1, 100]  # Weights for different components of the loss function
ITERATIONS = 10000  # Number of training iterations
OPTIMIZER = "adam"  # Optimizer for the first part of the training
BATCH_SIZE = 256  # Batch size


# Define Allen-Cahn PDE
def pde(X, u):
    du_t = dde.grad.jacobian(u, X, j=2)
    du_xx = dde.grad.hessian(u, X, i=0, j=0)
    du_yy = dde.grad.hessian(u, X, i=1, j=1)
    return du_t - EPSILON * (du_xx + du_yy) - 10 * (u - u**3)


# Define boundary conditions
def boundary_condition(X, on_boundary):
    return on_boundary


# Define initial condition
def boundary_initial(X, on_initial):
    _, _, t = X
    return on_initial and np.isclose(t, 0)  # Check if at the initial time


# Initialize a function for the temperature field
def init_func(X):
    x, y, _ = X[:, 0:1], X[:, 1:2], X[:, 2]
    sine_pattern = np.sin(np.pi * x) * np.sin(np.pi * y)
    return sine_pattern


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

# Predict the solution at each time point
Ts = []  # List to store the solution at each time point
for time in t:
    t_ = np.ones(
        (nelx + 1) * (nely + 1),
    ) * (time)
    X = np.column_stack((x_, y_))  # Making 2d array with x and y
    X = np.column_stack((X, t_))  # Making 3d array with the 2d array and t
    T = model.predict(X)  # Predict the solution
    T = T.reshape(
        T.shape[0],
    )
    T = T.reshape(nelx + 1, nely + 1)
    Ts.append(T)


# Function to plot the heatmap of the solution
def plotheatmap(T, time):
    time_text = round(time * delta_t, ndigits=2)
    plt.clf()  # Clear the current plot figure
    plt.title(f"Time = {time_text}     Surface: Dependent variable u ({time_text})")
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

# Save the animation as a mp4 file
anim.save("pinn_allenCahn_solution.mp4", writer="ffmpeg")
