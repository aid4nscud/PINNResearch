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
def pde(x, y, t, u, u_t):
    u_xx = dde.grad.hessian(u, x, y, t)[0, 0]
    u_yy = dde.grad.hessian(u, x, y, t)[1, 1]
    return u_t - alpha * (u_xx + u_yy)

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

# Define the custom PINN model
class CustomPINN(dde.maps.FNN):
    def forward(self, x):
        u = self.net(x)
        x, y, t = dde.split_dim(x, 3)
        u_x = dde.grad.jacobian(u, x)
        u_y = dde.grad.jacobian(u, y)
        u_t = dde.grad.jacobian(u, t)
        return u, u_x, u_y, u_t

# Optimizer and Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Using Adam optimizer


# Compile and Train Model
net = CustomPINN([2, 50, 50, 50, 50, 50, 1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile(optimizer, 1e-4)
model.train(iterations=10000, callbacks=[pde_resampler])

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

