%env KERAS_BACKEND=tensorflow


from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob

from keras.utils import np_utils


import cv2
from google.colab.patches import cv2_imshow

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test, 10)

Round = 30
floRound = float(Round)
sum = 0
for j in range(1,Round+1):
  model_2 = Sequential()
  model_2.add(Conv2D (input_shape=(28,28,1), filters=4, kernel_size=(3,3), padding='same', activation='relu'))
  model_2.add(Conv2D (filters=32, kernel_size=(3,3), padding='same', activation='relu'))
  model_2.add(Conv2D (filters=128, kernel_size=(3,3), padding='same', activation='relu'))
  model_2.add(MaxPooling2D(pool_size=(2,2)))
  model_2.add(Flatten())

  model_2.add(Dense(units=30, kernel_initializer='normal', activation='relu'))

  model_2.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))


  model_2.summary()
  model_2.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])

  model_2.fit(x_train, y_train, epochs=10, batch_size=1200, verbose=2)

  scores = model_2.evaluate(x_test, y_test)
  sum += float(scores[1]*100.0)
  print("\n\t[INFO] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))
  print()

average = sum / floRound
print("\t Average is = {:2.1f}%".format(average))
