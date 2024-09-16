#!pip install yfinance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import seaborn as sns
import tensorflow as tf
import keras_tuner as kt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras_tuner import Hyperband

from tensorflow.keras.models import Model
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam

# Get the stock ticker for NVIDIA
stock = yf.Ticker("NVDA")

# Get historical market data
data_hist = stock.history(start="2014-01-01", end="2023-12-31", actions=False, rounding=False)

# Set 'Date' as the index of the DataFrame
data_hist.reset_index(inplace=True)
data_hist.set_index('Date', inplace=True)

# Save the DataFrame to a CSV file without the default numerical index
data_hist.to_csv('stock_price_NVDA.csv', index=True)

# Load the data from the CSV, with 'Date' as the index
data = pd.read_csv('stock_price_NVDA.csv', index_col='Date', parse_dates=True)

# Convert 'Date' index to UTC timezone to handle timezone-aware datetime objects
data.index = pd.to_datetime(data.index, utc=True)

# Use the index directly to calculate days, months, and years
data['Days'] = (data.index - data.index.min()).days
data['Month'] = data.index.month
data['Year'] = data.index.year

# Splitting data into training and testing sets
split_size = int(0.8 * len(stock_prices))  # 80% for training
train_data, test_data = stock_prices.iloc[:split_size], stock_prices.iloc[split_size:]
y_train = train_data['Price']  # Define y_train as the target variable for the training set
y_test = test_data['Price']

# Naive forecast
naive_forecast = y_test.shift(1)

# Plotting
plt.figure(figsize=(9, 6))

# Plot training data
plt.plot(train_data.index, y_train, label='Train Data', color='blue')

# Plot testing data
plt.plot(test_data.index, y_test, label='Test Data', color='orange')

# Plot naive forecast - ensure to skip the first NaN value caused by the shift
plt.plot(test_data.index[1:], naive_forecast[1:], '.', label='Naive Forecast', color='purple')

plt.xlabel('Date')
plt.ylabel('NVIDIA Stock Price')
plt.legend()
plt.show()
