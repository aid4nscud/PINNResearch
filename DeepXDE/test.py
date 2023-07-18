import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

# Configuration Parameters
diff_coef = 1.0
plate_length = 1.0
t_final = 1.0
density_collocation = 20000
density_boundary = 10000

layer_size = [3] + [32] * 8 + [1]
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
learning_rate = 1e-3
iterations = 10000


def main():
    # Check GPU Availability
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

    # Define Domain
    geom = dde.geometry.Rectangle([0, 0], [plate_length, plate_length])
    timedomain = dde.geometry.TimeDomain(0, t_final)
    geotime = dde.geometry.GeometryXTime(geom, timedomain)

    # Define PDE
    def pde(X, u):
        du_X = tf.gradients(u, X)[0]
        du_x, du_y, du_t = du_X[:, 0:1], du_X[:, 1:2], du_X[:, 2:3]
        du_xx = tf.gradients(du_x, X)[0][:, 0:1]
        du_yy = tf.gradients(du_y, X)[0][:, 1:2]
        return du_t - diff_coef * (du_xx + du_yy)

    # Define Boundary Conditions
    def bc_hot(x, on_boundary):
        return np.logical_and(on_boundary, np.isclose(x[:, 0], plate_length))

    def bc_cold(x, on_boundary):
        return np.logical_and(on_boundary, ~np.logical_or(np.isclose(x[:, 0], 0), np.isclose(x[:, 0], plate_length)))

    bc_right_edge = dde.DirichletBC(geotime, lambda x: 100.0, bc_hot)
    bc_left = dde.NeumannBC(geotime, lambda x: 0.0, bc_cold)
    bc_top = dde.NeumannBC(geotime, lambda x: 0.0, bc_cold)
    bc_bottom = dde.NeumannBC(geotime, lambda x: 0.0, bc_cold)
    bcs = [bc_right_edge, bc_left, bc_top, bc_bottom]

    # Define Training Data
    data = dde.data.TimePDE(
        geotime, pde, bcs, num_domain=density_collocation, num_boundary=density_boundary)

    # Define Neural Network Architecture and Model
    net = dde.nn.FNN(layer_size, activation, initializer)
    model = dde.Model(data, net)
    model.compile(optimizer, learning_rate)

    # Train Model
    model.train(iterations=iterations)

    # Generate Test Data
    x_data = np.linspace(0, plate_length, num=100)
    y_data = np.linspace(0, plate_length, num=100)
    t_data = np.linspace(0, t_final, num=100)
    test_x, test_y, test_t = np.meshgrid(x_data, y_data, t_data)
    test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T

    # Predict Solution
    predicted_solution = model.predict(test_domain)
    residual = model.predict(test_domain, operator=pde)

    # Reshape Solution
    predicted_solution = predicted_solution.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])
    residual = residual.reshape(test_x.shape[0], test_y.shape[1], test_t.shape[2])

    # Plot and Save Solution
    animate_solution(predicted_solution, 'pinn_solution.gif', 'Heat equation solution', 'Temperature (K)', t_data)
    animate_solution(residual, 'pinn_residual.gif', 'Residual plot', 'Residual', t_data)


def animate_solution(data, filename, title, label, t_data):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(data[:, :, 0], origin='lower', cmap='hot', interpolation="bilinear")
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def updatefig(k):
        im.set_array(data[:, :, k])
        ax.set_title(f'{title}, t = {t_data[k]:.2f}')  # Update the title with current time step
        return [im]

    ani = animation.FuncAnimation(fig, updatefig, frames=range(data.shape[2]), interval=50, blit=True)
    ani.save(filename, writer='pillow')


if __name__ == "__main__":
    main()
