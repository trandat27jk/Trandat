import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class sig_moid_neural_network:
    def __init__(self, data, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2
        self.data = data.T
        self.weight_1 = None
        self.bias_1 = None
        self.weight_2 = None
        self.bias_2 = None

    def initialize(self):
        self.weight_1 = np.random.randn(self.node_1, len(self.data)) * 0.01
        self.bias_1 = np.zeros((self.node_1, 1)) * 0.01
        self.weight_2 = np.random.randn(self.node_2, self.node_1) * 0.01
        self.bias_2 = np.zeros((self.node_2, 1)) * 0.01
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

    def forward_propagation(self, data):
        z_1 = np.dot(self.weight_1, data) + self.bias_1
        a_1 = 1/(1+np.exp(-z_1))  # Sigmoid activation function
        z_2 = np.dot(self.weight_2, a_1) + self.bias_2
        a_2 = 1/(1+np.exp(-z_2))  # Sigmoid activation function
        return a_1, a_2

    def loss_function(self, y_hat, Y):
        m = len(y_hat)
        Y = Y.reshape(1, len(Y))
        J = -(1/m)*np.sum(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat))
        return J

    def training_data(self, X_train, Y_train, learning_rate, iterations):
        m = len(X_train)

        # Initialize weights and biases outside the loop
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.initialize()

        for i in range(iterations):
            # Forward propagation
            a_1, a_2 = self.forward_propagation(X_train)

            # Backward propagation
            d_Z_2 = a_2 - Y_train
            dw_2 = (1 / m) * np.dot(d_Z_2, a_1.T)
            db_2 = (1 / m) * np.sum(d_Z_2, axis=1, keepdims=True)

            d_Z_1 = np.dot(self.weight_2.T, d_Z_2) * a_1*(1 - a_1)
            dw_1 = (1 / m) * np.dot(d_Z_1, X_train.T)
            db_1 = (1 / m) * np.sum(d_Z_1, axis=1, keepdims=True)
            # loss function
            # print(a_2.shape, Y_train.shape)
            J = self.loss_function(a_2, Y_train)
            print(J)
            # Update weights and biases
            self.weight_1 = self.weight_1 - learning_rate * dw_1
            self.bias_1 = self.bias_1 - learning_rate * db_1
            self.weight_2 = self.weight_2 - learning_rate * dw_2
            self.bias_2 = self.bias_2 - learning_rate * db_2
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2, J

    def fit(self, X_train, Y_train, learning_rate, iterations):
        self.weight_1, self.bias_1, self.weight_2, self.bias_2, J = self.training_data(
            X_train, Y_train, learning_rate, iterations)
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2, J

    def predict(self, X_test):
    
        a_1, a_2 = self.forward_propagation(X_test.T)
        y_hat = a_2
        y_hat = np.round(y_hat)
        return y_hat

    def accuracy(self, y_hat, Y_test):
        y_hat = np.round(y_hat)
        accuracy = np.sum(y_hat == Y_test)/len(Y_test)
        return accuracy
