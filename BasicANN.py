# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:39:26 2016

@author: sebastien
Credit to fchollet for the original code :)
(@ https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)

ITT: Seb dicks around with a pointless neural network.
"""

import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Data GET!
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Setting up X-axes(?)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
plt.pcolor(X_train[0].reshape(28, 28))

# Setting up Y-axes(??)
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# And now for the fun stuff
model = Sequential()
model.add(Dense(256, input_shape=(784,)))
# (@ nb_epoch=5) 46.4s & acc:0.87 for 512, 28s & acc:0.84 for 256
model.add(Activation('relu'))  #works faster than sigmoid... and better, someohow???
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.layers
model.summary(line_length=100)

model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
%time model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, validation_data=(X_test, Y_test), verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
