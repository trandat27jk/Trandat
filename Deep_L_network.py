import numpy as np
import math

class DeepNetWork:
    def __init__(self, layers, list_nodes):
        self.layers = layers
        self.list_nodes = list_nodes
        self.param = {}
        self.grads = {}

    def initialize(self):
        for i in range(1, self.layers+1):
            self.param['W'+str(i)] = np.random.randn(self.list_nodes[i],
                                                     self.list_nodes[i-1]) * np.sqrt(2/self.list_nodes[i-1])
            self.param['b'+str(i)] = np.zeros((self.list_nodes[i], 1))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        return np.exp(z)/np.sum(np.exp(z), axis=0)

    def forward_propagation(self, data):
        self.param['A'+str(0)] = data

        for i in range(1, self.layers+1):

            self.param['Z'+str(i)] = np.dot(self.param['W'+str(i)],
                                            self.param['A'+str(i-1)]) + self.param['b'+str(i)]
            if i == self.layers:
                self.param['A'+str(i)] = self.softmax(self.param['Z'+str(i)])
            else:
                self.param['A'+str(i)] = self.relu(self.param['Z'+str(i)])
        return self.param['A'+str(self.layers)]

    def cross_entropy_function(self, y_hat, Y):
        m = Y.shape[1]
        cost = -np.sum(Y*np.log(y_hat))
        return cost

    def relu_derivative(self, Z):
        return np.where(Z <= 0, 0, 1)

    def backward_propagation(self, X_train, Y_train):
        m = X_train.shape[1]

        for i in range(self.layers, 0, -1):

            if i == self.layers:
                self.param['dZ'+str(i)] = self.param['A'+str(i)] - Y_train
                self.grads['dW'+str(i)] = (1/m)*np.dot(self.param['dZ' +
                                                                  str(i)], self.param['A'+str(i-1)].T)
                self.grads['db'+str(i)] = (1/m) * \
                    np.sum(self.param['dZ'+str(i)], axis=1, keepdims=True)

            else:
                self.param['dZ'+str(i)] = np.dot(self.param['W'+str(i+1)].T, self.param['dZ'+str(
                    i+1)]) * self.relu_derivative(self.param['Z'+str(i)])
                self.grads['dW'+str(i)] = (1/m)*np.dot(self.param['dZ' +
                                                                  str(i)], self.param['A'+str(i-1)].T)
                self.grads['db'+str(i)] = (1/m) * \
                    np.sum(self.param['dZ'+str(i)], axis=1, keepdims=True)

    def mini_batch(self,X, Y, mini_batch_size = 64, seed = 0):
        
        
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # number of training examples
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
        
        inc = mini_batch_size

        # Step 2 - Partition (shuffled_X, shuffled_Y).
        # Cases with a complete mini batch size only i.e each of 64 examples.
        num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            # (approx. 2 lines)
            # mini_batch_X =  
            # mini_batch_Y =
            # YOUR CODE STARTS HERE
            mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
            ### END CODE HERE ###
            
            # YOUR CODE ENDS HERE
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
        if m % mini_batch_size != 0:
            #(approx. 2 lines)
            # mini_batch_X =
            # mini_batch_Y =
            # YOUR CODE STARTS HERE
            mini_batch_X = shuffled_X[:, int(m/mini_batch_size)*mini_batch_size : ]
            mini_batch_Y = shuffled_Y[:, int(m/mini_batch_size)*mini_batch_size : ]
            
            # YOUR CODE ENDS HERE
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    def initialize_adam(self):
        v = {}
        s = {}
        for i in range(1, self.layers+1):
            v['dW'+str(i)] = np.zeros((self.list_nodes[i], self.list_nodes[i-1]))
            v['db'+str(i)] = np.zeros((self.list_nodes[i], 1))
            s['dW'+str(i)] = np.zeros((self.list_nodes[i], self.list_nodes[i-1]))
            s['db'+str(i)] = np.zeros((self.list_nodes[i], 1))
        return v, s

    def update_parameter_Adam(self, learning_rate, beta1, beta2, epsilon, t, v, s):
        v_corrected = {}
        s_corrected = {}

        for i in range(1, self.layers+1):
            v['dW'+str(i)] = beta1*v['dW'+str(i)] + \
                (1-beta1)*self.grads['dW'+str(i)]
            v['db'+str(i)] = beta1*v['db'+str(i)] + \
                (1-beta1)*self.grads['db'+str(i)]
            v_corrected['dW'+str(i)] = v['dW'+str(i)]/(1-np.power(beta1, t))
            v_corrected['db'+str(i)] = v['db'+str(i)]/(1-np.power(beta1, t))
            s['dW'+str(i)] = beta2*s['dW'+str(i)] + (1-beta2) * \
                np.square(self.grads['dW'+str(i)])
            s['db'+str(i)] = beta2*s['db'+str(i)] + (1-beta2) * \
                np.square(self.grads['db'+str(i)])
            s_corrected['dW'+str(i)] = s['dW'+str(i)]/(1-np.power(beta2, t))
            s_corrected['db'+str(i)] = s['db'+str(i)]/(1-np.power(beta2, t))
            self.param['W'+str(i)] = self.param['W'+str(i)] - learning_rate*(
                v_corrected['dW'+str(i)]/(np.sqrt(s_corrected['dW'+str(i)])+epsilon))
            self.param['b'+str(i)] = self.param['b'+str(i)] - learning_rate*(
                v_corrected['db'+str(i)]/(np.sqrt(s_corrected['db'+str(i)])+epsilon))
        return self.param

    def Adam_optimization(self, X_train, Y_train, learning_rate, iterations, batch_size, beta1, beta2, epsilon):
        self.initialize()
        m = X_train.shape[1]
        t = 0
        v, s = self.initialize_adam()
        for i in range(iterations):
            mini_batches = self.mini_batch(X_train, Y_train, batch_size)
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                A = self.forward_propagation(mini_batch_X)
                self.backward_propagation(mini_batch_X, mini_batch_Y)
                t = t + 1
                self.update_parameter_Adam(
                    learning_rate, beta1, beta2, epsilon, t, v, s)
            if i % 10 == 0:
                print("Cost after iteration {}: {}".format(
                    i, self.cross_entropy_function(A, mini_batch_Y)))

    def update_parameters(self, learning_rate):
        for i in range(1, self.layers+1):

            self.param['W'+str(i)] = self.param['W'+str(i)] - \
                learning_rate*self.grads['dW'+str(i)]

            self.param['b'+str(i)] = self.param['b'+str(i)] - \
                learning_rate*self.grads['db'+str(i)]

    def fit(self, X_train, Y_train, optimizer, learning_rate, iterations, batch_size, beta1,beta2,epsilon):
        self.initialize()

        if optimizer == "adam":
            
            self.Adam_optimization(
                X_train, Y_train, learning_rate, iterations, batch_size, beta1, beta2, epsilon)
        else:
            for i in range(iterations):
                A = self.forward_propagation(X_train)
                self.backward_propagation(X_train, Y_train)
                self.update_parameters(learning_rate)
                if i % 10 == 0:
                    print("Cost after iteration {}: {}".format(
                        i, self.cross_entropy_function(A, Y_train)))

    def predict(self, X_test):

        A = self.forward_propagation(X_test)
        real_value = np.argmax(A, axis=0)
        return A, real_value

    def accuracy(self, y_hat, Y_test):
        y_hat = np.argmax(y_hat, axis=0)
        Y_test = np.argmax(Y_test, axis=0)
        return (y_hat == Y_test).mean()
