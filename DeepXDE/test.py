import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Configuration Parameters
FLAGS.diff_coef = 1.0
FLAGS.plate_length = 1.0
FLAGS.t_final = 1.0
FLAGS.density_collocation = 20000
FLAGS.density_boundary = 10000
FLAGS.density_initial = 5000
FLAGS.density_test = 20000
FLAGS.layer_size = [3] + [32] * 8 + [1]
FLAGS.activation = "tanh"
FLAGS.initializer = "Glorot uniform"
FLAGS.optimizer = "adam"
FLAGS.learning_rate = 1e-3
FLAGS.iterations = 10000


def main():
    # Check GPU Availability
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

    # Define Domain
    geom = dde.geometry.Rectangle(0, 0, FLAGS.plate_length, FLAGS.plate_length)
    timedomain = dde.geometry.TimeDomain(0, FLAGS.t_final)
    geotime = dde.geometry.GeometryXTime(geom, timedomain)

    # Define PDE
    def pde(X, u):
        du_X = tf.gradients(u, X)[0]
        du_x, du_y, du_t = du_X[:, 0:1], du_X[:, 1:2], du_X[:, 2:3]
        du_xx = tf.gradients(du_x, X)[0][:, 0:1]
        du_yy = tf.gradients(du_y, X)[0][:, 1:2]
        return du_t - FLAGS.diff_coef * (du_xx + du_yy)

    # Define Boundary Conditions
    def bc_hot(x, on_boundary):
        return on_boundary and dde.is_close(x[0], FLAGS.plate_length)

    def bc_cold(x, on_boundary):
        return on_boundary and not (dde.is_close(x[0], 0) or dde.is_close(x[0], FLAGS.plate_length))

    # Define Training Data
    data = dde.data.TimePDE(
        geotime, pde, [bc_hot], num_domain=FLAGS.density_collocation, num_boundary=FLAGS.density_boundary)
    ic = dde.IC(geotime, func_ic, lambda _, on_initial: on_initial)

    # Define Neural Network Architecture and Model
    net = dde.nn.FNN(FLAGS.layer_size, FLAGS.activation, FLAGS.initializer)
    model = dde.Model(data, net, ic)
    model.compile(FLAGS.optimizer, FLAGS.learning_rate)

    # Train Model
    model.train(iterations=FLAGS.iterations)

    # Generate Test Data
    x_data = np.linspace(0, FLAGS.plate_length, num=100)
    y_data = np.linspace(0, FLAGS.plate_length, num=100)
    t_data = np.linspace(0, FLAGS.t_final, num=100)
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


def func_ic(x):
    return np.zeros((len(x), 1))


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
