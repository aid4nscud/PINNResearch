import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf


# Check if TensorFlow is using GPU
print("GPU DEVICES:\n\n")
print(tf.config.list_physical_devices('GPU'))
print("\n------------------------------------------\n")

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
def func_bc(x):
    # Apply condition for boundaries
    return np.where(
        np.logical_or(np.isclose(x[:, 0], length), np.isclose(x[:, 1], width)),
        100.0,
        0.0
    )[:, None]

def func_ic(x):
    # The initial condition is zero everywhere
    return np.zeros((len(x), 1))

bc = dde.DirichletBC(geotime, func_bc, lambda _, on_boundary: on_boundary)
ic = dde.IC(geotime, func_ic, lambda _, on_initial: on_initial)

# Training Data
data = dde.data.TimePDE(geotime, pde, [bc, ic], num_domain=4000, num_boundary=2000, num_initial=1000, num_test=1000)

# Model Architecture
layer_size = [3] + [32]*4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
learning_rate = 0.001

#Compile and Train Model
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
model.compile(optimizer, learning_rate)
model.train(10000)
model.compile("L-BFGS")

# Results
x_data = np.linspace(0, length, num=100)
y_data = np.linspace(0, width, num=100)
t_data = np.linspace(0, 1, num=100)
test_x, test_y, test_t = np.meshgrid(x_data, y_data, t_data)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T
predicted_solution = model.predict(test_domain)
residual = model.predict(test_domain, operator=pde)

# Reshape the data for animation
predicted_solution = predicted_solution.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])
residual = residual.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])

fig = plt.figure(figsize=(6, 6))
ax = plt.axes()

# Create the initial plot and color bar
cmap = ax.contourf(test_x[:, :, 0], test_y[:, :, 0], predicted_solution[:, :, 0], cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Predicted Solution, t={:.2f}'.format(t_data[0]))
colorbar = fig.colorbar(cmap, ax=ax)

# Update function for the animation
def update(i):
    ax.cla()  # Clear the current plot

    cmap = ax.contourf(test_x[:, :, i], test_y[:, :, i], predicted_solution[:, :, i], cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Predicted Solution, t={:.2f}'.format(t_data[i]))
    colorbar.update_bruteforce(cmap)
    fig.colorbar(cmap, cax=colorbar.ax)

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=test_t.shape[2], interval=200)

# Save the animation as an mp4 file
anim.save('heat2dPrediction.mp4', writer='ffmpeg')