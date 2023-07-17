import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

# Configuration Parameters
ALPHA = 1.0
LENGTH = 1.0
WIDTH = 1.0
MAX_TIME = 1.0
LAYER_SIZE = [3] + [32] * 8 + [1]
ACTIVATION = "tanh"
INITIALIZER = "Glorot uniform"
OPTIMIZER = "adam"
LEARNING_RATE = 1e-3
ITERATIONS = 10000

def main():
    # Check GPU Availability
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

    # Define Domain
    geom = dde.geometry.Rectangle([0, 0], [LENGTH, WIDTH])
    timedomain = dde.geometry.TimeDomain(0, MAX_TIME)
    geotime = dde.geometry.GeometryXTime(geom, timedomain)

    # Define PDE
    def pde(X, u):
        du_X = tf.gradients(u, X)[0]
        du_x, du_y, du_t = du_X[:, 0:1], du_X[:, 1:2], du_X[:, 2:3]
        du_xx = tf.gradients(du_x, X)[0][:, 0:1]
        du_yy = tf.gradients(du_y, X)[0][:, 1:2]
        return du_t - ALPHA * (du_xx + du_yy)
    
    def output_transform(x, y):
        # Get the individual components
        x_val, y_val, t_val = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        # Apply boundary conditions
        left = y * (1 - x_val)
        right = 100 * x_val
        top = y * (1 - y_val)
        bottom = y * y_val
        initial = y * t_val

        # Combine the conditions
        return left + right + top + bottom + initial

    # Define Training Data
    data = dde.data.TimePDE(geotime, pde, [],
                             num_domain=10000, num_boundary=2000, num_initial=4000, num_test=10000)
    pde_resampler = dde.callbacks.PDEPointResampler(period=50)

    # Define Neural Network Architecture and Model
    net = dde.nn.FNN(LAYER_SIZE, ACTIVATION, INITIALIZER)
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile(OPTIMIZER, LEARNING_RATE)

    # Train Model
    model.train(iterations=ITERATIONS, callbacks=[pde_resampler])

    # Generate Test Data
    x_data = np.linspace(0, LENGTH, num=100)
    y_data = np.linspace(0, WIDTH, num=100)
    t_data = np.linspace(0, 1, num=100)
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
