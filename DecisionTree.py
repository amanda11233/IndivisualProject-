
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from yahoo_fin import stock_info as si
from collections import deque
from sklearn import metrics

import datetime as dt
from yahoo_fin import stock_info as si
import os
import time
import random

plt.style.use("bmh")


#setting seeds
# np.random.seed(314)
# tf.random.set_seed(314)
# random.seed(314)


def load_data(ticker, shuffle=True, future_days=1, test_size=0.25,
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

    df = df.drop(['ticker'], 1)


    df['Prediction'] = df[['adjclose']].shift(-future_days)

    x_future = df.drop(['Prediction'], 1)[:-future_days]
    y_future = df['Prediction'][:-future_days]
    result['x_future'] = x_future
    result['y_future'] = y_future

    # create the feature data set (X) and convert it to numpy array and remove the last 'x' rows/days
    X = np.array(df.drop(['Prediction'], 1))[:-future_days]

    # create the target data set(y) and convert it to a numpy array and get all of the target values except the last days/rows
    y = np.array(df['Prediction'])[:-future_days]

    valid = df[X.shape[0]:]

    result['valid'] = valid

    result["X_train"], result["X_test"], result["y_train"], result["y_test"] =  train_test_split(X, y,
                                                                                                     test_size=0.25)

    return result

def create_model():
    model = DecisionTreeRegressor()
    return model


def predict_price(model, x_future):
    prediction = model.predict(x_future)
    return prediction

def get_mean_absolute_error(y_test, y_pred):
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return mae
def get_mean_squared_error(y_test, y_pred):
    mse = metrics.mean_squared_error(y_test, y_pred)

    return mse

def get_root_mean_squared_error(y_test, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    return rmse

def get_r_squared(y_test, y_pred):
    rsq = metrics.r2_score(y_test, y_pred)
    return rsq

ticker="TSLA"
FUTURE_DAYS = 30
TEST_SIZE = 0.25

data = load_data(ticker, shuffle=False, future_days=FUTURE_DAYS, test_size=TEST_SIZE,
          feature_columns=['adjclose', 'volume', 'open', 'high', 'low'])
model = create_model()
model = model.fit(data['X_train'], data['y_train'])


x_future = data['x_future'].tail(FUTURE_DAYS)
x_future = np.array(x_future)

prediction = predict_price(model, x_future)

y_future = data['y_future'].tail(FUTURE_DAYS)
y_future = np.array(y_future)



valid = data['valid']
valid['Predictions'] = prediction


print('Mean Absolute Error:', get_mean_absolute_error(y_future, prediction))
print('Mean Squared Error:', get_mean_squared_error(y_future, prediction))
print('Root Mean Squared Error:', get_root_mean_squared_error(y_future, prediction))
print('R2:', get_r_squared(y_future, prediction))

plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(data["df"]["close"][-200:])
plt.plot(valid[['close', "Predictions"]])
plt.legend(['Original', 'Valid', 'Predicted'])
plt.show()



