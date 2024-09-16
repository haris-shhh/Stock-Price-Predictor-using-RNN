import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import seaborn as sns

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

# Display the first few rows of the updated DataFrame to verify the changes
print(data.head())

# Visualise the historical price movement of the stock
stock_prices.plot(figsize=(10,7),color='red')
plt.ylabel('Nvidia Stock Price')
plt.xlabel('Year')
plt.title('Nvidia Stock Close Price 2014 - 2024')
plt.show()

# Create subplots for multiple features
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 12))

# Plot each price series
ax[0].plot(data.index, data['Open'], color='green', label='Open Price')
ax[0].plot(data.index, data['Close'], color='red', label='Close Price')
ax[0].legend()
ax[1].plot(data.index, data['High'], color='orange', label='High Price')
ax[1].plot(data.index, data['Low'], color='purple', label='Low Price')
ax[1].legend()

# Customize subplots
ax[0].set_title('Open and Close Prices of NVDA')
ax[0].set_ylabel('Price')
ax[1].set_title('High and Low Prices of NVDA')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Price')

# Show the plot
plt.tight_layout()
plt.show()

# Visualize the data with a pairplot
sns.pairplot(data)
plt.show()
     
