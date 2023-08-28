import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations in Python


ALPHA = dde.Variable(1.0)  # Initialize the value of alpha


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=0, j=1)
    return dy_t - ALPHA * (dy_xx + dy_yy)


# Load the data
data_dict = np.load("temperature_data.npz")
X_data = data_dict["x_data"]
Y_data = data_dict["y_data"]
T_data = data_dict["T_data"]

# Flatten and stack to create observation points
observe_x = np.stack(np.meshgrid(X_data.flatten(), Y_data.flatten(), 1), -1).reshape(
    -1, 3
)
observe_y = T_data.flatten()[:, None]

# Define the spatial and temporal domain
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define observation points
observe_y_bc = dde.icbc.PointSetBC(observe_x, observe_y, component=0)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_y_bc],
    num_domain=1000,
    num_boundary=500,
    num_initial=250,
    anchors=observe_x,
    num_test=10000,
)

layer_size = [3] + [60] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=ALPHA
)
variable = dde.callbacks.VariableValue(ALPHA, period=1000)
losshistory, train_state = model.train(iterations=50000, callbacks=[variable])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
plt.show()
plt.savefig("loss_history_inverseHeat2d")
plt.close()
