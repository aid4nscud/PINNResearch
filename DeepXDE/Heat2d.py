import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Some useful functions
START = 0
END_TIME = 1
ALPHA = 1.0


def pde(X, T):
    dT_xx = dde.grad.hessian(T, X, j=0)
    dT_yy = dde.grad.hessian(T, X, j=1)
    dT_t = dde.grad.jacobian(T, X, j=2)

    return dT_t - (ALPHA * (dT_xx + dT_yy))


def r_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(x, 1)


def l_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(x, 0)


def up_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(y, 1)


def down_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(y, 0)


def boundary_initial(X, on_initial):
    x, y, t = X
    return on_initial and np.isclose(t, 0)


def init_func(X):
    t = np.zeros((len(X), 1))
    return t


def dir_func_r(X):
    # IMPORTANT: The boundary condition is scaled down to be between 0 and 1 (1 for this scenario) so that the NN can minimize all of the diff losses effectively using Tanh.
    return np.ones((len(X), 1))


def func_zero(X):
    return np.zeros((len(X), 1))


num_domain = 30000
num_boundary = 8000
num_initial = 20000
layer_size = [3] + [60] * 5 + [1]
activation_func = "tanh"
initializer = "Glorot uniform"
lr = 1e-3
# Applying Loss weights as given below
# [PDE Loss, BC1 loss - Neumann Left , BC2 loss - Dirichlet Right, BC3 loss- Neumann up, BC4 loss - Neumann down, IC Loss]
loss_weights = [10, 1, 1, 1, 1, 10]
epochs = 10000
optimizer = "adam"
batch_size_ = 256

geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
timedomain = dde.geometry.TimeDomain(0, END_TIME)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_l = dde.NeumannBC(geomtime, func_zero, l_boundary)
bc_r = dde.DirichletBC(geomtime, dir_func_r, r_boundary)
bc_up = dde.NeumannBC(geomtime, func_zero, up_boundary)
bc_low = dde.NeumannBC(geomtime, func_zero, down_boundary)
ic = dde.IC(geomtime, init_func, boundary_initial)


data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_l, bc_r, bc_up, bc_low, ic],
    num_domain=num_domain,
    num_boundary=num_boundary,
    num_initial=num_initial,
    exclusions=[
        lambda X, on_boundary: on_boundary and np.isclose(X[0], 0) and np.isclose(X[1], 0),  # bottom left vertex
        lambda X, on_boundary: on_boundary and np.isclose(X[0], 1) and np.isclose(X[1], 0),  # bottom right vertex
        lambda X, on_boundary: on_boundary and np.isclose(X[0], 0) and np.isclose(X[1], 1),  # top left vertex
        lambda X, on_boundary: on_boundary and np.isclose(X[0], 1) and np.isclose(X[1], 1),  # top right vertex
    ],
)


net = dde.maps.FNN(layer_size, activation_func, initializer)
net.apply_output_transform(lambda x, y: abs(y))
model = dde.Model(data, net)

model.compile(optimizer, lr=lr, loss_weights=loss_weights)

losshistory, trainstate = model.train(
    epochs=epochs,
    batch_size=batch_size_,
)
model.compile("L-BFGS-B")
dde.optimizers.set_LBFGS_options(
    maxcor=50,
)
losshistory, train_state = model.train(epochs=epochs, batch_size=batch_size_)
dde.saveplot(losshistory, trainstate, issave=True, isplot=True)

fig, ax = plt.subplots()

ax = fig.add_subplot(111)
nelx = 100
nely = 100
timesteps = 101
x = np.linspace(0, 1, nelx + 1)
y = np.linspace(0, 1, nely + 1)
t = np.linspace(0, 1, timesteps)
delta_t = t[1] - t[0]
xx, yy = np.meshgrid(x, y)


x_ = np.zeros(shape=((nelx + 1) * (nely + 1),))
y_ = np.zeros(shape=((nelx + 1) * (nely + 1),))
for c1, ycor in enumerate(y):
    for c2, xcor in enumerate(x):
        x_[c1 * (nelx + 1) + c2] = xcor
        y_[c1 * (nelx + 1) + c2] = ycor
Ts = []

for time in t:
    t_ = np.ones(
        (nelx + 1) * (nely + 1),
    ) * (time)
    X = np.column_stack((x_, y_))
    X = np.column_stack((X, t_))
    T = model.predict(X)
    T = T * 100
    T = T.reshape(
        T.shape[0],
    )
    T = T.reshape(nelx + 1, nely + 1)
    Ts.append(T)


def plotheatmap(T, time):
    # Clear the current plot figure
    plt.clf()
    plt.title(f"Temperature at t = {round(time*delta_t, ndigits=2)} unit time")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolor(xx, yy, T, cmap="jet")
    plt.colorbar()
    return plt


def animate(k):
    plotheatmap(Ts[k], t[k])


anim = animation.FuncAnimation(
    plt.figure(), animate, interval=1, frames=len(t), repeat=False
)

anim.save("pinn_solution.gif")
