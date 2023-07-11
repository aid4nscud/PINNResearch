
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0
length = 1.0
max_time = 1.0

# Computational Domain
geom = dde.geometry.Interval(0, length)
timedomain = dde.geometry.TimeDomain(0,max_time)
geotime = dde.geometry.GeometryXTime(geom, timedomain)

# PDE Residual
def pde(x, y):
    dy_t = dde.grad.jacobian(y , x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return dy_t - dy_xx * alpha

# Initial Condidtion & Boundary Condition
def func_bc(x):
    # Apply condition for left and right boundaries
    return np.where(np.isclose(x[:, 0], length), 100.0, 0.0)[:, None]

def func_ic(x):
    # The initial condition is zero everywhere
    return np.zeros((len(x), 1))

bc = dde.DirichletBC(geotime, func_bc, lambda _, on_boundary: on_boundary)
ic = dde.IC(geotime, func_ic, lambda _, on_initial: on_initial)

# Training Data
data = dde.data.TimePDE(geotime, pde, [bc, ic], num_domain=4000, num_boundary=2000, num_initial=1000, num_test=1000)

# Model Architecture
layer_size = [2] + [32]*3 + [1]
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
losshistory, train_state = model.train()

# Results
x_data = np.linspace(0, length, num = 100)
t_data = np.linspace(0, 1, num=100)
test_x, test_t = np.meshgrid(x_data,t_data)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_t))).T
predicted_solution = model.predict(test_domain)
residual = model.predict(test_domain, operator=pde)

# Reshape the predicted_solution array
predicted_solution = predicted_solution.reshape(test_x.shape)
residual = residual.reshape(test_x.shape)


# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the predicted solution in the first subplot
cmap1 = axs[0].pcolormesh(test_x, test_t, predicted_solution, shading='auto')
axs[0].set_xlabel('x')
axs[0].set_ylabel('t')
axs[0].set_title('Predicted Solution')
colorbar1 = fig.colorbar(cmap1, ax=axs[0])
colorbar1.set_label('Solution')

# Plot the residual in the second subplot
cmap2 = axs[1].pcolormesh(test_x, test_t, residual, shading='auto')
axs[1].set_xlabel('x')
axs[1].set_ylabel('t')
axs[1].set_title('Residual')
colorbar2 = fig.colorbar(cmap2, ax=axs[1])
colorbar2.set_label('Residual')

# Adjust the spacing between subplots
fig.tight_layout()

# Show the figure with both subplots
plt.show()
