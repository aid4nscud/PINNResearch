import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Load the saved model
net = dde.maps.FNN([3] + [60] * 5 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: abs(y))
model = dde.Model(net=net)
model.compile("adam", lr=1e-3)
model.restore("model_path")

# Set up animation
nelx = 100
nely = 100
timesteps = 101
x = np.linspace(0, 1, nelx + 1)
y = np.linspace(0, 1, nely + 1)
t = np.linspace(0, 1, timesteps)
delta_t = t[1] - t[0]
xx, yy = np.meshgrid(x, y)

def animate(k):
    x_ = np.zeros(shape=((nelx + 1) * (nely + 1),))
    y_ = np.zeros(shape=((nelx + 1) * (nely + 1),))
    for c1, ycor in enumerate(y):
        for c2, xcor in enumerate(x):
            x_[c1 * (nelx + 1) + c2] = xcor
            y_[c1 * (nelx + 1) + c2] = ycor
    t_ = np.ones((nelx + 1) * (nely + 1)) * t[k]
    X = np.column_stack((x_, y_, t_))
    T = model.predict(X)
    T = T * 100
    T = T.reshape(nelx + 1, nely + 1)
    plt.clf()
    plt.title(f"Temperature at t = {round(t[k]*delta_t, ndigits=2)} unit time")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolor(xx, yy, T, cmap="jet")
    plt.colorbar()

anim = animation.FuncAnimation(plt.figure(), animate, frames=len(t), interval=1, repeat=False)
anim.save("pinn_solution.gif")
