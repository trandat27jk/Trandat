import tensorflow as tf
from tensorflow.keras.datasets import mnist
from Deep_L_network import DeepNetWork
import numpy as np
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
original_y_test=y_test
#one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10).T
y_test = tf.keras.utils.to_categorical(y_test, 10).T


# Reshape the training and test set
x_train = x_train.reshape(60000, 784).T
x_test = x_test.reshape(10000, 784).T

# Normalize the training and test set
x_train = x_train / 255
x_test = x_test / 255

model=DeepNetWork(5,[x_train.shape[0],25,25,25,25,10])
model.fit(x_train, y_train, optimizer="adam", learning_rate=0.01, iterations=100, batch_size=512, beta1=0.9, beta2=0.999, epsilon=1e-8)

y_pred,value=model.predict(x_test)

#accuracy
print("accuracy:",np.sum(np.argmax(y_pred,axis=0)==original_y_test)/len(original_y_test))