import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    num_domain=6000,
    num_boundary=200,
    num_initial=400,
    num_test=6000,
)
pde_resampler = dde.callbacks.PDEPointResampler(period=50)

# Model Architecture
layer_size = [3] + [50] * 8 + [1]
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "L-BFGS-B"
dde.optimizers.config.set_LBFGS_options(maxcor=10, ftol=1.0e-6, gtol=1.0e-6, maxiter=15000, maxfun=15000)
learning_rate = 0.001

# Compile and Train Model
net = dde.nn.FNN(layer_size, activation, initializer)
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


# Reshape the data for animation
predicted_solution = predicted_solution.reshape(
    test_x.shape[0], test_y.shape[1], test_t.shape[2]
)
residual = residual.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])

# Prepare the figure

fig, ax = plt.subplots(figsize=(6, 6))

# Calculate the global maximum temperature across all frames
global_max = np.max(predicted_solution)

cmap = plt.get_cmap("hot")
norm = plt.Normalize(vmin=0, vmax=global_max)

# Hold the contour plot in a dictionary
plots = {}
plots["contour"] = ax.contourf(test_x[:, :, 0], test_y[:, :, 0], predicted_solution[:, :, 0], cmap=cmap, norm=norm)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Predicted Solution, t={:.2f}".format(t_data[0]))

colorbar = fig.colorbar(plots["contour"], ax=ax)
cax = colorbar.ax

def update(i):
    # Remove the previous contours
    for coll in plots["contour"].collections:
        coll.remove()
    # Create new contours
    plots["contour"] = ax.contourf(test_x[:, :, i], test_y[:, :, i], predicted_solution[:, :, i], cmap=cmap, norm=norm)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predicted Solution, t={:.2f}".format(t_data[i]))

anim = animation.FuncAnimation(fig, update, frames=test_t.shape[2], interval=200)
# Save the animation as an mp4 file
anim.save("heat2DPrediction.mp4", writer="ffmpeg")
