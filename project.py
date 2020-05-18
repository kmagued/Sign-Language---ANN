import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from neuralNetwork import NeuralNetwork
from layer import Layer
from activation_layer import ActivationLayer
from activations import sigmoid, sigmoid_derivative, relu, relu_derivative
from losses import mse, mse_prime
from preprocess import preprocess

import keras
from keras.models import Sequential
from keras.layers import Dense

#Reading csv files for training and testing
sign_lang_test = pd.read_csv('sign_mnist_test.csv')
sign_lang = pd.read_csv('sign_mnist_train.csv')

#-----------------------------Data Preprocessing-----------------------------------#
sign_lang_y = sign_lang['label']
sign_lang_x = sign_lang.drop('label', axis=1)

test_y = sign_lang_test['label']
test_x = sign_lang_test.drop('label', axis=1)

#Training data
X = preprocess(sign_lang_x)

#Testing data
X_test = preprocess(test_x)

#One hot encoding labels
y = np.zeros((sign_lang_y.shape[0], sign_lang_y.max()+1))
y[np.arange(sign_lang_y.shape[0]), sign_lang_y] = 1

y_test = np.zeros((test_y.shape[0], test_y.max()+1))
y_test[np.arange(test_y.shape[0]), test_y] = 1

#------------------------------------Model-----------------------------------------#
# nn = NeuralNetwork()
# l1 = Layer(X.shape[1], 54)
# l2 = Layer(54, y.shape[1])

# nn.add(l1)
# nn.add(ActivationLayer(relu, relu_derivative))
# nn.add(l2)
# nn.add(ActivationLayer(sigmoid, sigmoid_derivative))

#Training the model
# nn.use(mse, mse_prime)
# nn.fit(X, y, 100, 0.2)

model = Sequential()
model.add(Dense(54, input_dim=X.shape[1], activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=1000, batch_size=128)

model.save('model.h5')
del model

#Prediction phase
# out = nn.predict(X_test)
# out = model.predict(X_test)
# pred = list()
# for i in range(len(out)):
#     pred.append(np.argmax(out[i]))

# test = list()
# for i in range(len(y_test)):
#     test.append(np.argmax(y_test[i]))

#Accuracy
# a = accuracy_score(test, pred)
# print(f'Accuracy: {round(a*100, 1)}%')

#Saving weights and biases
# np.save('weights1.npy', l1.weights)
# np.save('bias1.npy', l1.bias)

# np.save('weights2.npy', l2.weights)
# np.save('bias2.npy', l2.bias)