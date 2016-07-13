# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:28:23 2016

@author: sebastien

**UNFINISHED**

"""
# ...importing prereqs
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import Recurrent
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(1337)
train, test, dicts = pickle.load(open("atis.pkl",'rb'),encoding='bytes')

plt.pcolormesh(test)


# ...data stuff

model = Sequential()
# ...actual RNN

