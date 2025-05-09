import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ["DenseNetwork"]

def activation(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_activation(x):
    fx = activation(x)
    return fx * (1 - fx)

def compute_loss(weights, biases, inputs, targets):
    loss = 0
    for x, y in zip(inputs, targets):
        _, activations = forward(weights, biases, x)
        output_layer = len(activations) - 1
        loss += 0.5 * (activations[output_layer] - y) ** 2
    return float(loss / len(inputs))

def init_params(layer_dims):
    weights = {}
    biases = {}
    for layer in range(1, len(layer_dims)):
        weights[layer] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * np.sqrt(2.0 / layer_dims[layer - 1])
        biases[layer] = np.zeros((layer_dims[layer], 1))
    return weights, biases

def forward(weights, biases, input_vec):
    z_values = {}
    a_values = {}
    a_values[0] = input_vec.reshape(-1, 1)
    for layer in range(1, len(weights) + 1):
        z_values[layer] = weights[layer] @ a_values[layer - 1] + biases[layer]
        a_values[layer] = activation(z_values[layer])
    return z_values, a_values

class DenseNetwork:
    def __init__(self, architecture=[4, 8, 1]):
        self.architecture = architecture
        self.weights, self.biases = init_params(self.architecture)

    def train(self, X, y, lr=0.01, n_epochs=50):
        self.loss_history = [compute_loss(self.weights, self.biases, X, y)]
        print(f"Initial Loss = {self.loss_history[0]}")
        last_layer = len(self.architecture) - 1

        for epoch in range(n_epochs):
            for x_sample, y_sample in zip(X, y):
                z_vals, activations = forward(self.weights, self.biases, x_sample)
                grads = {}

                grads[last_layer] = (activations[last_layer] - y_sample) * d_activation(z_vals[last_layer])

                for l in range(last_layer - 1, 0, -1):
                    grads[l] = (self.weights[l + 1].T @ grads[l + 1]) * d_activation(z_vals[l])

                for l in range(1, last_layer + 1):
                    self.weights[l] -= lr * grads[l] @ activations[l - 1].T
                    self.biases[l] -= lr * grads[l]

            self.loss_history.append(compute_loss(self.weights, self.biases, X, y))
            print(f"Epoch {epoch + 1} Loss = {self.loss_history[-1]}")

    def predict(self, input_sample):
        _, activations = forward(self.weights, self.biases, input_sample)
        return activations[len(self.architecture) - 1][0][0]