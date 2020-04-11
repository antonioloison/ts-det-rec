"""Neural Network class. This is a simple implementation of a neural network using numpy arrays."""

import numpy as np
import math
from math_utils import *

# CONSTANTES
FIRST_LEARNING_RATE = 100
# si il est trop lent changer jusqu'a ce qu'a diverge
NUMBER_OF_EPOCHS = 10000


class Nnetwork:

    def __init__(self, layers, functions):
        self.layers = layers  # the first layer must have as many neurons as inputs and the last layer must have as many neuron as classifications
        self.number_of_layers = len(self.layers)
        self.values = list(np.zeros([layers[i], 1]) for i in range(len(layers)))
        self.functions = functions
        self.activations = [np.zeros([layers[0], 1])] + list(
            self.functions[i](np.zeros([layers[i + 1], 1])) for i in range(len(layers) - 1))
        self.derivatives = list(derivative(self.functions[i]) for i in range(len(layers) - 1))
        self.weights = list(np.random.rand(layers[i], layers[i - 1]) * 0.01 for i in range(1, len(layers)))
        self.bias = list(np.zeros([layers[i], 1]) for i in range(1, len(layers)))

    def calculate(self, input_vector):
        """
        Computes the total cost function at the output of the NN for one image input
        :param input_vector: one dimension numpy array representing the neural network input
        :return: output vector (numpy array)
        """

        a = input_vector
        for i in range(1, self.number_of_layers):
            w = self.weights[i - 1]
            b = self.bias[i - 1]
            z = np.dot(w, a) + b
            a = self.functions[i - 1](z)

        return a

    def forward_propagation(self, input_matrix, expected_output):
        """
        Computes the total cost function for a batch of input images
        :param input_matrix: matrix where each column corresponds to one image vector
        :param expected_output: label vector
        :return: float of the total cost
        """

        m = np.shape(input_matrix)
        A = input_matrix
        self.activations[0] = A
        for i in range(1, self.number_of_layers):
            W = self.weights[i - 1]

            B = np.repeat(self.bias[i - 1], m[1], axis=1)

            Z = np.dot(W, A) + B
            self.values[i] = Z
            A = self.functions[i - 1](Z)
            self.activations[i] = A

        J = 1 / m[1] * np.sum(cost(A, expected_output))
        return A, J

    def accuracy(self, input, output):
        """
        Computes the neural network on a batch of images
        :param input: matrix of inputs
        :param output: expected outputs vector (labels)
        :return: accuracy percentage
        TO REFACTOR WITH MORE EFFICIENT NUMPY FUNCTIONS
        """
        score = 0
        A = self.forward_propagation(input, output)[0]
        for i in range(input.shape[1]):
            argmaxout, argmaxA = 0, 0
            maxout, maxA = 0, 0
            for j in range(3):
                if output[j, i] > maxout:
                    maxout = output[j, i]
                    argmaxout = j

                if A[j, i] > maxA:
                    argmaxA = j
                    maxA = A[j, i]
            if argmaxA == argmaxout:
                score += 1
        return score / A.shape[1] * 100

    def backward_propagation(self, input_matrix, output_matrix, expected_output, learning_rate=FIRST_LEARNING_RATE):
        """
        Updates neural network weights and biases by computing the backpropagation of the gradient
        :param input_matrix: matrix where each column corresponds to one image vector
        :param output_matrix: output of the neural network
        :param expected_output: array of the labels
        :param learning_rate: float
        :return: None
        """
        m = np.shape(input_matrix)
        dA = (1 / m[1]) * cost_derivative(output_matrix, expected_output)

        for i in reversed(range(1, self.number_of_layers)):
            dZ = dA * self.derivatives[i - 1](self.values[i])

            if i > 1:
                dW = (1 / m[1]) * np.dot(dZ, (self.activations[i - 1]).T)
            else:
                dW = (1 / m[1]) * np.dot(dZ, (input_matrix).T)

            dB = (1 / m[1]) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.weights[i - 1].T, dZ)

            self.weights[i - 1] = self.weights[i - 1] - learning_rate * dW
            self.bias[i - 1] = self.bias[i - 1] - learning_rate * dB


    def epochs(self, input_matrix, expected_output, number_of_epochs=NUMBER_OF_EPOCHS):
        """
        Performs gradient descent on neural network
        :param input_matrix: matrix where each column corresponds to one image vector
        :param expected_output: array of the labels
        :param number_of_epochs: integer
        :return: None
        """
        for i in range(number_of_epochs):

            (A, J) = self.forward_propagation(input_matrix, expected_output)

            self.backward_propagation(input_matrix, A, expected_output)
            if i % 50 == 0:
                print("Epoch " + str(i) + ": Cost -> " + str(J))

    def epoch_with_dev_set(self, input_matrix, expected_output, dev_input, dev_output):
        """
        Trains it until the accuracy gets to a stable minimum of the dev training
        :param input_matrix: matrix where each column corresponds to one image vector
        :param expected_output: array of the labels
        :param dev_input: dev set input
        :param dev_output: dev set expected output
        :return: None
        """
        i = 0
        results = [2, 3, 4, 5]
        decreasing_learning_rate = FIRST_LEARNING_RATE
        while results != sorted(results)[::-1] or results[-1] < 95:
            Acc = self.accuracy(dev_input, dev_output)
            (A, J) = self.forward_propagation(input_matrix, expected_output)
            if i % 10 == 0:
                print("Epoch " + str(i) + ": Cost -> " + str(J))
                print("This NN has an accuracy of " + str(Acc) + "%")
            if results[-1] - results[-2] > 10:
                decreasing_learning_rate /= 10
            self.backward_propagation(input_matrix, A, expected_output, learning_rate=decreasing_learning_rate)
            i += 1
            results.append(Acc)
            x = results.pop(0)

    def save(self, directory, nn_name):
        """
        Saves weights
        :param directory: string path of the direcotry
        :param nn_name: string of the nn name
        :return: None
        """
        np.save(directory + '\\' + 'saved_weights' + nn_name + '.npy', self.weights)
        np.save(directory + '\\' + 'saved_bias' + nn_name + '.npy', self.bias)

    def load(self, weights_path, bias_path):
        """
        Updates weight according to saved weights and biases
        :param weights_path: string path of weights
        :param bias_path: string path of biases
        :return: None
        """
        self.weights = np.load(weights_path, allow_pickle=True)
        self.bias = np.load(bias_path, allow_pickle=True)


# test forward_propagation
# NN1 = Nnetwork([3,2,1],[sigmoid,sigmoid])
# NN1.weights=[np.array([[0.25,0.5,0.3],[0.4,0.1,0.8]]),np.array([[0.2,0.7]])]
# print(NN1.forward_propagation(np.array([[0.7,0.2],[0.8,0.1],[0.9,0.3]]),np.array([[1,0]])))
# print(NN1.weights)
# print(NN1.bias)
# print(NN1.activations)
"""result

A,J = (array([[0.65981036, 0.62637652]]), 0.700154766241492)
W = [array([[0.25, 0.5 , 0.3 ],
       [0.4 , 0.1 , 0.8 ]]), array([[0.2, 0.7]])]
b = [array([[0.],
       [0.]]), array([[0.]])]
all A = [array([[0.],
       [0.],
       [0.]]), array([[0.69951723, 0.54735762],
       [0.74649398, 0.58175938]]), array([[0.65981036, 0.62637652]])]
"""

# test back_propagation
# NN1 = Nnetwork([3,2,1],[relu,sigmoid])
# NN1.weights=[np.array([[0.25,0.5,0.3],[0.4,0.1,0.8]]),np.array([[0.2,0.7]])]
# NN1.epoch1(np.array([[0.7,0.2,0.1],[0.8,0.1,0.2],[0.9,0.3,0.4]]),np.array([[1,0,0]]),number_of_epochs=600)
# print()1
# result
# [[0.5]
#  [0.5]]
# [[0.5 0.5]]
# [[0]]
# [[0.5]]
# [[0.62245933]]
# 0.47407698418010663
# [[-1.60653066]]
# [[-0.18877033 -0.18877033]]
# [[-0.37754067]]
# [[-0.37754067]]
# [[-0.18877033]
#  [-0.18877033]]
# [[0.68877033 0.68877033]]
# [[0.37754067]]


# test epoch
# NN1 = Nnetwork([3, 2, 1], [sigmoid, sigmoid])
# print(NN1.weights)
# print(NN1.bias)
# print(NN1.activations)
# NN1.epoch1(np.array([[0.7, 0.2], [0.8, 0.1], [0.9, 0.3]]), np.array([[1, 0]]))
