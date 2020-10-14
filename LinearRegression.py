
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
import matplotlib.dates as mdates
import datetime as dt
from yahoo_fin import stock_info as si
import os
import time
import random

plt.style.use("bmh")

# Import matplotlib package for date plots
import matplotlib.dates as mdates

#setting seeds
# np.random.seed(314)
# tf.random.set_seed(314)
# random.seed(314)


def load_data(ticker, shuffle=True, future_days=1, test_size=0.25,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):

    if(isinstance(ticker, str)):
        data = si.get_data(ticker)

    elif isinstance(ticker, pd.DataFrame):
        data = ticker

    else:
        raise TypeError("ticker can be either a str or a `pd.Dataframe`")

    df = pd.DataFrame(data, columns=['close'])
    df=df.reset_index()
    print(df.head())
    # Check data types in columns
    print(df.info())
    # Check for missing values in the columns
    print(df.isna().values.any())


    # Split data into train and test set: 80% / 20%
    train, test = train_test_split(df, test_size=0.20)

    return train, test, df


def create_model():
    model = LinearRegression()
    return model


def predict_price(model, x_future):
    prediction = model.predict(x_future)
    return prediction

def get_mean_absolute_error(y_test, y_pred):
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return mae

def get_r_squared(y_test, y_pred):
    rsq = metrics.r2_score(y_test, y_pred)
    return rsq

ticker="TSLA"
FUTURE_DAYS = 30
TEST_SIZE = 0.25

train, test, df = load_data(ticker, shuffle=False, future_days=FUTURE_DAYS, test_size=TEST_SIZE,
          feature_columns=['adjclose', 'volume', 'open', 'high', 'low'])


# Reshape index column to 2D array for .fit() method
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['close']



# Create LinearRegression Object
model = LinearRegression()
# Fit linear model using the train data set
model.fit(X_train, y_train)




# Create test arrays
X_test = np.array(test.index).reshape(-1, 1)
y_test = test['close']
print(y_test)
# Generate array with predicted values
y_pred = model.predict(X_test)
df['Prediction'] = model.predict(np.array(df.index).reshape(-1, 1))



# Generate 25 random numbers
randints = np.random.randint(2581, size=25)

# Select row numbers == random numbers
df_sample = df[df.index.isin(randints)]

print(df_sample.head())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R2: ', metrics.r2_score(y_test, y_pred))

# Create subplots to plot graph and control axes

# Train set graph
plt.figure(1, figsize=(16,10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()




