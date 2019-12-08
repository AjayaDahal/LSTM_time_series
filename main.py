import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import math

np.random.seed(7)
fileName = "WiFiData1Time"
sampleSize = 400
y = np.loadtxt(fileName + "yValues.txt")
x1 = np.loadtxt(fileName + str(sampleSize) + "RealFFT.txt")
x2 = np.loadtxt(fileName + str(sampleSize) + "ComplexFFT.txt")
sampleSize *= 2
xSize = sampleSize



x = np.ones((np.shape(x1)[0], xSize))
y = np.pad(y, (0, 7), 'constant')
x[:, 0:xSize:2] = x1
x[:, 1:xSize:2] = x2


scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)
#x_test =scaler.fit_transform(x_test)


x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

print(x_train)
print()
print(y_train)


x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
#x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
#y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))


#y_train = scaler.fit_transform(y_train)

#y_test = scaler.fit_transform(y_test)



model = Sequential()
model.add(LSTM(1, activation = 'relu', input_shape=(1, 800)))
#model.add(LSTM(1, activation='sigmoid'))
model.add(Dense(1))
model.compile(loss='mse',  optimizer='adam')
model.summary()


history_train = model.fit(x_train, y_train, epochs=300, verbose=2)

results_train = model.predict(x_train, y_train)
results_test = model.predict(x_test)


print(results_train)
print(y_train)

plt.plot(y_train, color="r", label='train')
plt.plot(results_train, color="g", label='results')
plt.show()
plt.plot(y_test, color="black", label='test')
plt.plot(results_test, color="blue", label='results')
plt.show()

plt.plot(history_train.history['loss'])
plt.show()

results_train = scaler.inverse_transform(results_train)
results_test = scaler.inverse_transform([results_test])
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], results_train[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], results_test[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))


