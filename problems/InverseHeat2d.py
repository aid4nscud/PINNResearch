import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Initialize the value of alpha
ALPHA = dde.Variable(1.0)
WIDTH = 1
LENGTH = 1
T_END = 1
BATCH_SIZE = 256
ITERATIONS = 10000


# Define the PDE
def pde(X, T):
    # Calculate second derivatives (Hessians) of T with respect to X in both dimensions
    dT_xx = dde.grad.hessian(T, X, i=0, j=0)
    dT_yy = dde.grad.hessian(T, X, i=1, j=1)

    # Calculate first derivative (Jacobian) of T with respect to X in time dimension
    dT_t = dde.grad.jacobian(T, X, j=2)

    # Return the defined PDE
    return dT_t - (ALPHA * dT_xx + dT_yy)



# Define geometry and time domains
geom = dde.geometry.Rectangle([0, 0], [WIDTH, LENGTH])  # Geometry domain
timedomain = dde.geometry.TimeDomain(0, T_END)  # Time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain)  # Space-time domain

# Load the data
data_dict = np.load("temperature_data.npz")
X_data = data_dict["x_data"]
Y_data = data_dict["y_data"]
T_data = data_dict["T_data"]

# Flatten and stack to create observation points
observe_x = np.hstack(
    (
        X_data.flatten()[:, None],
        Y_data.flatten()[:, None],
        np.ones_like(X_data.flatten())[:, None],
    )
)
observe_y = T_data.flatten()[:, None]

# Normalize all data
x_min = observe_x[:, 0].min()
x_max = observe_x[:, 0].max()
observe_x[:, 0] = (observe_x[:, 0] - x_min) / (x_max - x_min)

y_min = observe_x[:, 1].min()
y_max = observe_x[:, 1].max()
observe_x[:, 1] = (observe_x[:, 1] - y_min) / (y_max - y_min)

T_min = observe_y.min()
T_max = observe_y.max()
observe_y = (observe_y - T_min) / (T_max - T_min)

# Define observation points
observed_data = dde.PointSetBC(observe_x, observe_y)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [observed_data],
    num_domain=1000,
    num_boundary=500,
    num_initial=250,
    anchors=observe_x,
    num_test=10000,
)

# Network architecture
layer_size = [3] + [60] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

# Compile model
model.compile(
    "adam",
    lr=0.001,
    external_trainable_variables=[ALPHA],
)

# Define callback to save ALPHA
variable = dde.callbacks.VariableValue(ALPHA, period=1000)

# Train the model
losshistory, train_state = model.train(
    iterations=ITERATIONS, batch_size=BATCH_SIZE, callbacks=[variable]
)

# Save and plot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
plt.show()
plt.savefig("loss_history_plot_inverseHeat2d")
plt.close()
