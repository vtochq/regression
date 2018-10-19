import pandas as pd
#import my_lib
import datetime
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from time import time

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate, Activation, LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset[0]) - window_size - 1):
        data_X.append(dataset[:, i:(i + window_size)])
        data_Y.append(dataset[0, i + window_size])
    return(np.array(data_X), np.array(data_Y))

data_raw = pd.read_csv('dash_price.csv', header=0, encoding='utf8', sep=',')[::-1]
#dataset = data_raw['Close'].values
dataset = np.array((data_raw['Open'].values, data_raw['Close'].values, data_raw['High'].values, data_raw['Low'].values,\
 pd.to_numeric(data_raw['Volume'].str.replace(',', '').values)/500000, pd.to_numeric(data_raw['Market Cap'].str.replace(',', '').values)/10000000))

dataset = dataset[:4]

feature_count = dataset.shape[0]
window_size = 10
test_size = round(len(dataset[0])*0.014)

plt.figure(figsize = (15, 5))
plt.plot(dataset[0], label = "Open price")
plt.plot(dataset[1], label = "Close price")
plt.plot(dataset[2], label = "High price")
plt.plot(dataset[3], label = "Low price")
#plt.plot(dataset[4], label = "Volume")
#plt.plot(dataset[5], label = "Market Cap")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Dash price")
plt.legend()
plt.show()

scaler = MinMaxScaler(feature_range = (0, 1))
#dataset = scaler.fit_transform(dataset)
dataset[0]=scaler.fit_transform(dataset[0].reshape(-1,1))[:,0]


#data_train, data_test = train_test_split(dataset, test_size=.1, shuffle=False)

data_train = dataset[:, :-test_size]
data_test = dataset[:, -test_size:]

train_X, train_Y = create_dataset(data_train, window_size)
test_X, test_Y = create_dataset(data_test, window_size)
print("Original training data shape:")
print(train_X.shape)

#train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
#test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
#print("New training data shape:")
#print(train_X.shape)

def fit_model(train_X, train_Y, window_size = 1):
    model = Sequential()
    model.add(LSTM(window_size, input_shape = (feature_count, window_size), return_sequences = True))
    model.add(Dropout(0.1))
    model.add(LSTM(256))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss = "mean_squared_error", optimizer = "adam")
    model.summary()
    model.fit(train_X,
              train_Y,
              epochs = 5,
              batch_size = 1,
              verbose = 1,
              validation_split = 0.1)
    return(model)

model1 = fit_model(train_X, train_Y, window_size)


def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)

# Start with training predictions.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
#train_predict_plot[0, window_size:len(train_predict) + window_size] = train_predict
#train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict
train_predict_plot[0, window_size:len(train_predict) + window_size] = train_predict[:,0]

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[0, len(train_predict) + (window_size * 2) + 1:len(dataset[0]) - 1] = test_predict[:,0]

# Create the plot.
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset[0].reshape(-1,1)), label = "True Open value")
plt.plot(train_predict_plot[0], label = "Training set prediction")
plt.plot(test_predict_plot[0], label = "Test set prediction")
plt.xlabel("Days")
plt.ylabel("Dash price in USD")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()
