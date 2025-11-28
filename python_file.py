import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

np.random.seed(42)

import pandas as pd
df = pd.read_csv("scatter-points.csv")
points = df.to_numpy()

blue_points = points[points[:,0] < 0.4]
blue_points = blue_points[blue_points[:,1] > 0.6]
mask = ~np.isin(points, blue_points).all(axis=1)
red_points = points[mask]

X = np.concatenate((red_points, blue_points), axis=0)
# Red = 0, Blue = 1 (Binary Classification)
Y_red = np.zeros((len(red_points), 1))
Y_blue = np.ones((len(blue_points), 1))
Y = np.concatenate((Y_red, Y_blue), axis=0)

# Shuffle data for better SGD training
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X[indices]
Y_shuffled = Y[indices].flatten() # Flatten Y to be a vector of scalars for single output

# --- 2. Test-Train Split ---
# Total number of samples: 106
total_samples = len(X_shuffled)
# Use 80% for training
train_size = int(0.8 * total_samples) 

# Split the data
X_train, X_test = X_shuffled[:train_size], X_shuffled[train_size:]
Y_train, Y_test = Y_shuffled[:train_size], Y_shuffled[train_size:]

print(f"Total Samples: {total_samples}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")


def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(a):
    """Derivative of the sigmoid function, where a = sigmoid(z)."""
    return a * (1.0 - a)

def tanh(z):
    """Hyperbolic tangent activation function."""
    return np.tanh(z)

def d_tanh(a):
    """Derivative of the tanh function, where a = tanh(z)."""
    return 1.0 - a**2


class NeuralNetwork:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.weights = [
            np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0/layer_dims[i])
            for i in range(self.num_layers)
        ]
        self.biases = [np.zeros(layer_dims[i+1]) for i in range(self.num_layers)]

    def forward(self, x):
        a = x
        self.zs = []
        self.activations = [a]
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.zs.append(z)
            if i == self.num_layers - 1:
                a = sigmoid(z)
            else:
                a = tanh(z)
            self.activations.append(a)
        return a

    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

    def backward(self, x, y_true, learning_rate=0.01):
        y_pred = self.activations[-1]
        delta = y_pred - y_true
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        for l in reversed(range(self.num_layers)):
            grads_w[l] = np.outer(self.activations[l], delta)
            grads_b[l] = delta
            if l > 0:
                delta = np.dot(self.weights[l], delta) * d_tanh(self.activations[l])
        for l in range(self.num_layers):
            self.weights[l] -= learning_rate * grads_w[l]
            self.biases[l] -= learning_rate * grads_b[l]
        return y_pred

    def fit(self, X, Y, epochs=500, learning_rate=0.01, print_interval=50):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(X, Y):
                self.forward(x)
                self.backward(x, y_true, learning_rate)
                loss = self.compute_loss(y_true, self.activations[-1])
                total_loss += loss
            avg_loss = total_loss / len(X)
            if (epoch % print_interval == 0) or (epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
            yield self.weights, self.biases  


layer_dims = [2, 5, 1]
nn = NeuralNetwork(layer_dims)


fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(blue_points[:,0], blue_points[:,1], color='blue', label='Class 1')
ax.scatter(red_points[:,0], red_points[:,1], color='red', label='Class 0')
ax.set_xlim(-0.1, 1.15)
ax.set_ylim(-0.1, 1.15)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

lines = [ax.plot([], [], label=f'Neuron {i+1}')[0] for i in range(layer_dims[1])]
x_vals = np.linspace(-0.1, 1.15, 200)

def update(params):
    weights, biases = params
    w = weights[0]  
    b = biases[0]  
    for i, line in enumerate(lines):
        w1, w2 = w[:, i]
        bias = b[i]
        if abs(w2) < 1e-8:
            y_plot = np.full_like(x_vals, -bias)
        else:
            m = -w1 / w2
            c = -bias / w2
            y_plot = m * x_vals + c
        line.set_data(x_vals, y_plot)
    return lines

ani = animation.FuncAnimation(
    fig, update,
    frames=nn.fit(X_shuffled, Y_shuffled, epochs=500, learning_rate=0.1),
    interval=60, blit=True, repeat=False, cache_frame_data=False
)

plt.show()