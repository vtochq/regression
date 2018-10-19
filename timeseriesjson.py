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

import matplotlib.pyplot as plt
import json


def create_dataset(dataset, window_size = 1, col=0):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        data_X.append(dataset[i:(i + window_size), :])
        data_Y.append(dataset[i + window_size, col])
    return(np.array(data_X), np.array(data_Y))


def fit_model(train_X, train_Y, window_size = 1):
    model = Sequential()
    model.add(Dense(window_size, input_shape = (window_size, ))) # , kernel_initializer='normal'
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss = "mean_squared_error", optimizer = "Nadam", metrics=['accuracy'])
    model.summary()
    model.fit(train_X[:,:,0], #[:,:,0]
              train_Y,
              epochs = epoch,
              batch_size = batch_size,
              verbose = 1) # , validation_split = 0.1
    return(model)


def fit_model_ltsm(train_X, train_Y, window_size = 1):
    model = Sequential()
    model.add(LSTM(window_size, input_shape = (window_size, feature_count))) #, return_sequences = True
    #model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss = "mean_squared_error", optimizer = "Nadam", metrics=['accuracy'])
    model.summary()
    model.fit(train_X, #[:,:,0]
              train_Y,
              epochs = epoch,
              batch_size = batch_size,
              verbose = 1) # , validation_split = 0.1
    return(model)


def predict_and_score(model, X, Y, col=0):
    # Make predictions on the original scale of the data.
    pred = np.zeros((len(X),feature_count))
    pred[:, col] = model.predict(X)[:,0]
    pred = scaler.inverse_transform(pred)[:, col]
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = np.zeros((len(X),feature_count))
    orig_data[:, col] = Y
    orig_data = scaler.inverse_transform(orig_data)[:, col]
    #orig_data = Y
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data, pred))
    return(score, pred)

def future_predict(predict_to = 10):
    future = np.array([dataset[-window_size:]])
    for i in range(0, predict_to):
        Y = model1.predict(np.array([future[i,:,0]]))
        print(Y)
        #future = np.vstack((future, np.array([np.vstack((future[-1][-47:],(0,0,Y,0,0,0,0)))])))
        future = np.vstack((future, np.array([np.vstack((future[-1][-window_size+1:],(Y)))])))
    future_plot = np.empty((len(dataset)+predict_to+1))
    future_plot[:] = np.nan
    future_plot[len(dataset):len(dataset)+predict_to+1] = scaler.inverse_transform(future[:,-1:,col])[:,-1]
    plt.figure(figsize = (15, 5))
    plt.plot(scaler.inverse_transform(dataset)[:,col], label = "True Open value")
    plt.plot(future_plot, label = "Future prediction")
    plt.xlabel("Days")
    plt.ylabel("Dash price in BTC")
    plt.title("Comparison true vs. predicted training / test")
    plt.legend()
    plt.show()

with open('dash.json') as f:
    data_raw = json.load(f)

dataset = np.zeros((len(data_raw),7))
for i in range(0,len(data_raw)):
    dataset[i]=[data_raw[i]['high'], data_raw[i]['low'], data_raw[i]['open'], data_raw[i]['close'],\
    data_raw[i]['volume']/10000, data_raw[i]['quoteVolume']/150000, data_raw[i]['weightedAverage']]

dataset = dataset[:,2].reshape(-1,1)

feature_count = dataset.shape[1]

col=0
window_size = 240
batch_size = 1
test_size = round(len(dataset)*0.05)
epoch = 50
Test = 0

print("Dataset size: ",len(data_raw), ". Feature count: ", feature_count, ". Windows size: ", window_size,". Test size: ", test_size)

'''
plt.figure(figsize = (15, 5))
plt.plot(dataset[:,2], label = "Open price")
plt.plot(dataset[:,3], label = "Close price")
plt.plot(dataset[:,0], label = "High price")
plt.plot(dataset[:,1], label = "Low price")
plt.plot(dataset[:,4], label = "Volume")
plt.plot(dataset[:,5], label = "quoteVolume")
plt.plot(dataset[:,6], label = "wAvg")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Dash price")
plt.legend()
plt.show()
'''

scaler = MinMaxScaler(feature_range = (0, 1))
dataset=scaler.fit_transform(dataset)


if Test:
    data_train = dataset[:-test_size, :]
    data_test = dataset[-test_size:, :]
    test_X, test_Y = create_dataset(data_test, window_size, col)
else:
    data_train = dataset

train_X, train_Y = create_dataset(data_train, window_size, col)

print("Training data shape: ", train_X.shape)

#model1 = fit_model(train_X, train_Y, window_size)
model1 = fit_model_ltsm(train_X, train_Y, window_size)

rmse_train, train_predict = predict_and_score(model1, train_X, train_Y, col)
print("Training data score: %.2f RMSE" % rmse_train)

if Test:
    rmse_test, test_predict = predict_and_score(model1, test_X, test_Y, col)
    print("Test data score: %.2f RMSE" % rmse_test)


# Start with training predictions.
train_predict_plot = np.empty((len(dataset)))
train_predict_plot[:] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size] = train_predict

# Add test predictions.
if Test:
    test_predict_plot = np.empty((len(dataset)))
    test_predict_plot[:] = np.nan
    test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1] = test_predict

# Create the plot.
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset)[:,col], label = "True Open value")
plt.plot(train_predict_plot, label = "Training set prediction")
if Test:
    plt.plot(test_predict_plot, label = "Test set prediction")
plt.xlabel("Days")
plt.ylabel("Dash price in BTC")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()


future_predict(2)
