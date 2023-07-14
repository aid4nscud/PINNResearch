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
def pde(x, u, ux, uy, ut):
    u_xx = dde.grad.hessian(u, x)[0, 0]
    u_yy = dde.grad.hessian(u, x)[1, 1]
    return ut - alpha * (u_xx + u_yy)

# Initial Condition & Boundary Condition
def func_bc_right_edge(x, on_boundary):
    return on_boundary and np.isclose(x[0], length)

def func_ic(x):
    # The initial condition is zero everywhere
    return np.zeros((len(x), 1))

def func_zero(x):
    return np.zeros_like(x)

bc_right_edge = dde.DirichletBC(
    geotime, lambda x: 100, func_bc_right_edge
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
    num_domain=5060,
    num_boundary=160,
    num_initial=320,
    num_test=5060,
)

pde_resampler = dde.callbacks.PDEPointResampler(period=50)

# Optimizer and Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Using Adam optimizer

# Compile and Train Model
net = dde.maps.FNN([3] + [50] * 5 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile(optimizer, loss_weights=[1e-4], check_numerics=True)
model.train(iterations=100000, callbacks=[pde_resampler])

# Results
x_data = np.linspace(0, length, num=100)
y_data = np.linspace(0, width, num=100)
t_data = np.linspace(0, 1, num=100)
test_x, test_y, test_t = np.meshgrid(x_data, y_data, t_data)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T
predicted_solution = model.predict(test_domain)

# Reshape the data for animation
predicted_solution = predicted_solution.reshape((len(x_data), len(y_data), len(t_data)))

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

def update(i):
    # Remove the previous plots
    for coll in plots["contour"].collections:
        coll.remove()
    # Create new plots
    plots["contour"] = ax.contourf(test_x[:, :, i], test_y[:, :, i], predicted_solution[:, :, i], cmap=cmap, norm=norm)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predicted Solution, t={:.2f}".format(t_data[i]))

anim = animation.FuncAnimation(fig, update, frames=test_t.shape[2], interval=200)
# Save the animation as a GIF file
anim.save("heat2DPrediction.gif", writer=PillowWriter(fps=5))

