"""Simple math functions needed for the neural network"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(np.zeros(x.shape), x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu_derivative(x):
    m = x.shape
    A = np.zeros(m)
    return np.maximum(A, x)

def derivative(fct):
    if fct == sigmoid:
        return sigmoid_derivative
    elif fct == relu:
        return relu_derivative
    else:
        return "Error"

def cost(x, y):
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))

def cost_basique(x, y):
    return (x - y) ** 2

def cost_derivative_basique(x, y):
    return 2 * (x - y)

def cost_derivative(x, y):
    return (x - y) / (x * (1 - x))
