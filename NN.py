import numpy as np
import tensorflow as tf

# Parameters
alpha = 0.1
# now we have input dimension 3, because our input is (x, y, t)
layers = [3, 20, 20, 20, 20, 1]

# Define model architecture


class PDENet(tf.keras.Model):
    def __init__(self, layers):
        super(PDENet, self).__init__()
        self.layers = layers
        # Initialize the weights and biases for each layer in the neural network
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            # Each weight matrix's dimension is determined by the number of neurons in the current and next layers.
            weight = self.add_weight(
                shape=(layers[i], layers[i+1]), initializer='random_normal')
            # Each bias vector's dimension is determined by the number of neurons in the next layer.
            bias = self.add_weight(shape=(layers[i+1],), initializer='zeros')
            self.weights.append(weight)
            self.biases.append(bias)

    # Define the forward pass
    def call(self, X):
        Z = X
        # Perform matrix multiplication followed by addition of bias for all but the last layer, followed by an activation function (tanh)
        for i in range(len(self.layers)-2):
            Z = tf.nn.tanh(
                tf.add(tf.matmul(Z, self.weights[i]), self.biases[i]))
        # For the last layer, we only perform matrix multiplication and addition of bias, without an activation function
        Z = tf.add(tf.matmul(Z, self.weights[-1]), self.biases[-1])
        return Z


# Instantiate the model
model = PDENet(layers)

# Define MSE loss
def loss_fn(model, t, x, y):
    # We use TensorFlow's GradientTape for automatic differentiation.
    # The "persistent=True" argument allows us to compute multiple derivatives, as the tape is not immediately discarded after use.
    with tf.GradientTape(persistent=True) as tape:
        # We tell the tape to watch the t, x, y variables, as we want to differentiate with respect to these variables later.
        tape.watch(t)
        tape.watch(x)
        tape.watch(y)
        # Concatenate the t, x, y tensors and feed them into the model to compute the predicted temperature.
        # The model's call function performs the forward pass and computes the predicted temperature at (t, x, y).
        u = model(tf.concat([t, x, y], 1))
        # The tape.gradient method computes the derivative of a target (u) with respect to some source (t, x, y).
        # This gives us the first order derivatives of u with respect to t, x, and y.
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
    # After the tape is used to compute the first order derivatives, we compute the second order derivatives of u with respect to x and y.
    # This is needed because the heat equation involves second order spatial derivatives.
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    # We can now discard the tape with the "del" statement.
    del tape
    # The loss function is defined as the mean squared error between the left-hand side and right-hand side of the heat equation.
    # The left-hand side is the temporal derivative u_t and the right-hand side is the sum of the second order spatial derivatives, weighted by alpha.
    # We want this discrepancy to be as close to zero as possible for the given data points. This is what the model will try to accomplish during training.
    loss = tf.reduce_mean(tf.square(u_t - alpha*(u_xx + u_yy)))
    return loss


# Define training step


def train_step(model, optimizer, t, x, y):
    # Compute the loss and gradients with respect to model parameters
    with tf.GradientTape() as tape:
        loss = loss_fn(model, t, x, y)
    grads = tape.gradient(loss, model.trainable_variables)
    # Update model parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# Training data (random points for x, y, and t)
N_train = 1000
t_train = np.random.rand(N_train, 1)
x_train = np.random.rand(N_train, 1)
y_train = np.random.rand(N_train, 1)

# Train the model
epochs = 2000
optimizer = tf.keras.optimizers.Adam(lr=0.01)
for epoch in range(epochs):
    loss = train_step(model, optimizer, t_train, x_train, y_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
