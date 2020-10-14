import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yahoo_fin import stock_info as si
from collections import deque
from sklearn import metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random


#setting seeds
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):

    if(isinstance(ticker, str)):
        df = si.get_data(ticker)

    elif isinstance(ticker, pd.DataFrame):
        df = ticker

    else:
        raise TypeError("ticker can be either a str or a `pd.Dataframe`")

    result = {}

    result['df'] = df.copy()

    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe"

    if scale:
        column_scaler = {}

        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] =scaler
            result["column_scaler"]=column_scaler

        df["future"] = df["adjclose"].shift(-lookup_step)

        last_sequence = np.array(df[feature_columns].tail(lookup_step))

        df.dropna(inplace=True)

        sequence_data=[]
        sequences=deque(maxlen=n_steps)
        for entry, target in zip(df[feature_columns].values, df["future"].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])


        last_sequence = list(sequences) + list(last_sequence)
        last_sequence = np.array(last_sequence)

        result['last_sequence'] = last_sequence
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

        result["X_train"], result["X_test"], result["y_train"], result["y_test"] =  train_test_split(X, y,

                                                                          test_size=test_size, shuffle=shuffle)

        return result


def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error",
                 optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
                else:
                    model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))

            elif i == n_layers - 1:
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))

            else:
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))

        model.add(Dropout(dropout))

        model.add(Dense(1, activation="linear"))

        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

        return model



N_STEPS = 70
LOOKUP_STEP=30
TEST_SIZE =0.2
FEATURE_COLUMNS=["adjclose", "volume", "open", "high", "low"]
date_now = time.strftime("%Y-%m-%d")
N_LAYERS=3
CELL=LSTM
UNITS=256
DROPOUT=0.4
BIDIRECTIONAL=False
LOSS="huber_loss"
OPTIMIZER="adam"
BATCH_SIZE=64
EPOCHS=1
ticker="TSLA"
ticker_data_filename=os.path.join("data",f"{ticker}_{date_now}.csv")
model_name=f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-" \
           f"{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
       model_name += "-b"

    # create these folders if they does not exist
if not os.path.isdir("results"):
       os.mkdir("results")
if not os.path.isdir("logs"):
       os.mkdir("logs")
if not os.path.isdir("data"):
       os.mkdir("data")

# load the data
data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
# save the dataframe
data["df"].to_csv(ticker_data_filename)
# # construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
                                   save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
history = model.fit(data["X_train"], data["y_train"],
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=(data["X_test"], data["y_test"]),
                       callbacks=[checkpointer, tensorboard],
                       verbose=1)
model.save(os.path.join("results", model_name) + ".h5")


data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)
# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)

def predict(model, data):
    last_sequence = data["last_sequence"][-N_STEPS:]
    column_scaler = data["column_scaler"]
    last_sequence  = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction=model.predict(last_sequence)
    predicted_price=column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price

future_price = predict(model, data)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")

def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP],
                      y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP],
                      y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)

accuracy = get_accuracy(model, data)

print(accuracy)

def get_r2(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(
        map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(
        map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return metrics.r2_score(y_test, y_pred)

print(get_r2(model, data))

def plot_graph(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

plot_graph(model, data)

days=[]

def set_days(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))


    k=0
    for i in y_test:
        days.append(k)
        k+=1


def set_actual(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

    return y_test[-200:]

def set_pred(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

    return y_pred[-200:]


def get_days():
    set_days(model, data)
    return days

def get_actual():
    return set_actual(model, data)


def get_pred():
    return set_pred(model, data)


