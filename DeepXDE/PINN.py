import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import tensorflow as tf

# Parameters
alpha = 1.0
length = 1.0
width = 1.0
max_time = 1.0

# Computational Domain
geom = dde.geometry.Rectangle([0, 0], [length, width])
timedomain = dde.geometry.TimeDomain(0, max_time)
geotime = dde.geometry.GeometryXTime(geom, timedomain)

# PDE Residual
def pde(X, u):
    du_X = tf.gradients(u, X)[0]
    du_x, du_y, du_t = du_X[:, 0:1], du_X[:, 1:2], du_X[:, 2:3]
    du_xx = tf.gradients(du_x, X)[0][:, 0:1]
    du_yy = tf.gradients(du_y, X)[0][:, 1:2]
    return du_t - alpha * (du_xx + du_yy)

# Initial Condition & Boundary Condition
def func_bc_right_edge(x):
    # Assign a value of 100.0 if the point lies on the right edge and 0.0 otherwise
    return np.where(np.isclose(x[:, 0], length), 100.0, 0.0)[:, None]

def func_ic(x):
    # The initial condition is zero everywhere
    return np.zeros((len(x), 1))

def func_zero(x):
    return np.zeros_like(x)

bc_right_edge = dde.DirichletBC(
    geotime, func_bc_right_edge, lambda _, on_boundary: on_boundary
)
bc_left = dde.NeumannBC(
    geotime, func_zero, lambda x, on_boundary: on_boundary and np.isclose(x[0], 0) and not np.isclose(x[1], width) and not np.isclose(x[1], 0)
)
bc_top = dde.NeumannBC(
    geotime, func_zero, lambda x, on_boundary: on_boundary and np.isclose(x[1], width) and not np.isclose(x[0], length) and not np.isclose(x[0], 0)
)
bc_bottom = dde.NeumannBC(
    geotime, func_zero, lambda x, on_boundary: on_boundary and np.isclose(x[1], 0) and not np.isclose(x[0], length) and not np.isclose(x[0], 0)
)
ic = dde.IC(geotime, func_ic, lambda _, on_initial: on_initial)

# Training Data
data = dde.data.TimePDE(
    geotime,
    pde,
    [bc_right_edge, bc_left, bc_top, bc_bottom, ic],
    num_domain=5060,
    num_boundary=160,
    num_initial=320,
    num_test=5060,
)
pde_resampler = dde.callbacks.PDEPointResampler(period=50)

# Model Architecture
layer_size = [3] + [50] * 8 + [1]  # Same as in the provided code
activation = "tanh"  # Same as in the provided code
initializer = "Glorot uniform"  # Same as in the provided code

# Optimizer and Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Using Adam optimizer
learning_rate = None  # Not used for Adam optimizer

# Compile and Train Model
net = dde.maps.FNN(layer_size, activation, initializer)  # FNN with specified layer size and activation function
model = dde.Model(data, net)
model.compile(optimizer, learning_rate)
model.train(iterations=50000, callbacks=[pde_resampler])



# Results
x_data = np.linspace(0, length, num=100)
y_data = np.linspace(0, width, num=100)
t_data = np.linspace(0, 1, num=100)
test_x, test_y, test_t = np.meshgrid(x_data, y_data, t_data)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T
predicted_solution = model.predict(test_domain)
residual = model.predict(test_domain, operator=pde)

# Calculate the real solution using FDM
# Modify the code below to implement the Finite Difference Method and calculate the real solution
# real_solution = ...

# Calculate the error
error = np.abs(predicted_solution - real_solution)

# Reshape the data for animation
predicted_solution = predicted_solution.reshape(
    test_x.shape[0], test_y.shape[1], test_t.shape[2]
)
residual = residual.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])
error = error.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])

# Prepare the figure for predicted solution plot
fig, ax = plt.subplots(figsize=(6, 6))
global_max = np.max(predicted_solution)
cmap = plt.get_cmap("hot")
norm = plt.Normalize(vmin=0, vmax=global_max)
plots = {}
plots["contour"] = ax.contourf(test_x[:, :, 0], test_y[:, :, 0], predicted_solution[:, :, 0], cmap=cmap, norm=norm)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Predicted Solution, t={:.2f}".format(t_data[0]))
colorbar = fig.colorbar(plots["contour"], ax=ax)
cax = colorbar.ax

# Prepare the figure for error plot
fig_error, ax_error = plt.subplots(figsize=(6, 6))
error_plot = ax_error.contourf(test_x[:, :, 0], test_y[:, :, 0], error[:, :, 0], cmap=cmap)
ax_error.set_xlabel("x")
ax_error.set_ylabel("y")
ax_error.set_title("Error, t={:.2f}".format(t_data[0]))
colorbar_error = fig_error.colorbar(error_plot, ax=ax_error)
cax_error = colorbar_error.ax

def update(i):
    # Remove the previous plots
    for coll in plots["contour"].collections:
        coll.remove()
    for coll in error_plot.collections:
        coll.remove()
    # Create new plots
    plots["contour"] = ax.contourf(test_x[:, :, i], test_y[:, :, i], predicted_solution[:, :, i], cmap=cmap, norm=norm)
    error_plot = ax_error.contourf(test_x[:, :, i], test_y[:, :, i], error[:, :, i], cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predicted Solution, t={:.2f}".format(t_data[i]))
    ax_error.set_xlabel("x")
    ax_error.set_ylabel("y")
    ax_error.set_title("Error, t={:.2f}".format(t_data[i]))

anim = animation.FuncAnimation(fig, update, frames=test_t.shape[2], interval=200)
# Save the animation as a GIF file
anim.save("heat2DPrediction.gif", writer=PillowWriter(fps=5))

# Calculate the maximum error
max_error = np.max(error)

def update_error(i):
    # Remove the previous error plot
    for coll in error_plot.collections:
        coll.remove()
    # Create new error plot
    error_plot = ax_error.contourf(test_x[:, :, i], test_y[:, :, i], error[:, :, i], cmap=cmap)
    ax_error.set_xlabel("x")
    ax_error.set_ylabel("y")
    ax_error.set_title("Error, t={:.2f}".format(t_data[i]))

anim_error = animation.FuncAnimation(fig_error, update_error, frames=test_t.shape[2], interval=200)
# Save the error animation as a GIF file
anim_error.save("heat2DError.gif", writer=PillowWriter(fps=5))