import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

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


# Initial Condidtion & Boundary Condition
def func_bc_right_edge(x):
    # Assign a value of 100.0 if the point lies on the right edge and 0.0 otherwise.
    # The output is reshaped to ensure compatibility with TensorFlow.
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
    geotime, func_zero, lambda x, on_boundary: on_boundary and np.isclose(x[0], 0)
)
bc_top = dde.NeumannBC(
    geotime, func_zero, lambda x, on_boundary: on_boundary and np.isclose(x[1], width)
)
bc_bottom = dde.NeumannBC(
    geotime, func_zero, lambda x, on_boundary: on_boundary and np.isclose(x[1], 0)
)
ic = dde.IC(geotime, func_ic, lambda _, on_initial: on_initial)

# Training Data
data = dde.data.TimePDE(
    geotime,
    pde,
    [bc_right_edge, bc_left, bc_top, bc_bottom, ic],
    num_domain=10000,
    num_boundary=1000,
    num_initial=2000,
    num_test=10000,
)
pde_resampler = dde.callbacks.PDEPointResampler(period=50)

# Model Architecture
layer_size = [3] + [40] * 8 + [1]
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
learning_rate = 1e-3

# Compile and Train Model
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
model.compile(optimizer, learning_rate)
model.train(iterations=10000, callbacks=[pde_resampler])



# Results
x_data = np.linspace(0, length, num=100)
y_data = np.linspace(0, width, num=100)
t_data = np.linspace(0, 1, num=100)
test_x, test_y, test_t = np.meshgrid(x_data, y_data, t_data)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T
predicted_solution = model.predict(test_domain)
residual = model.predict(test_domain, operator=pde)


# Reshape the data for animation
predicted_solution = predicted_solution.reshape(
    test_x.shape[0], test_y.shape[1], test_t.shape[2]
)
residual = residual.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])

# Plot solution
fig1, ax1 = plt.subplots(figsize=(7, 7))
im1 = ax1.imshow(predicted_solution[:, :, 0], origin='lower', cmap='hot', interpolation="bilinear")
plt.colorbar(im1, ax=ax1, label="Temperature (K)")
ax1.set_title('Heat equation solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Define animation update function for solution
def updatefig1(k):
    im1.set_array(predicted_solution[:, :, k])
    return [im1]

# Create animation for solution
ani1 = animation.FuncAnimation(fig1, updatefig1, frames=range(test_t.shape[2]), interval=50, blit=True)

# Save solution as gif
ani1.save('pinn_solution.gif', writer='pillow')


# Plot residuals
fig2, ax2 = plt.subplots(figsize=(7, 7))
im2 = ax2.imshow(residual[:, :, 0], origin='lower', cmap='hot', interpolation="bilinear")
plt.colorbar(im2, ax=ax2, label="Residual")
ax2.set_title('Residual plot')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Define animation update function for residual
def updatefig2(k):
    im2.set_array(residual[:, :, k])
    return [im2]

# Create animation for residual
ani2 = animation.FuncAnimation(fig2, updatefig2, frames=range(test_t.shape[2]), interval=50, blit=True)
ani2.save('pinn_residual.gif', writer='pillow')
