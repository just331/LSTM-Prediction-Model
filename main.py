import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

rcParams['figure.figsize'] = 20, 10

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import MinMaxScaler

# Apple Historical Data from July 25th, 2020 to July 26, 2021
df = pd.read_csv('data/HistoricalData_1627345793457.csv')

# Preprocessing to just use first two columns in csv file
df = df[['Date', 'Close/Last']]

print(df.head())

print(df.dtypes)

# Get rid of the '$' from the 'Close/Last' column
df = df.replace({'\$': ''}, regex=True)

# Convert 'Close/Last' data type to float
df = df.astype({'Close/Last': float})
df['Date'] = pd.to_datetime(df.Date, format="%m/%d/%Y")

print(df.head())
print(df.dtypes)

# Define df's index value as 'Date' Column
df.index = df['Date']

# Line chart for data we have thus far
plt.plot(df['Close/Last'], label='Close Price History')

# Data Preparation
df = df.sort_index(ascending=True, axis=0)

data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close/Last'])

for i in range(0, len(data)):
    data['Date'][i] = df['Date'][i]
    data['Close/Last'][i] = df['Close/Last'][i]

print(data.head())


# Min-Max Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

data.index = data.Date
data.drop('Date', axis=1, inplace=True)

final_data = data.values
train_data = final_data[0:200, :]
valid_data = final_data[0:200, :]


scaled_data = scaler.fit_transform(final_data)
x_train_data, y_train_data = [],[]
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i-60: i, 0])
    y_train_data.append(scaled_data[i, 0])


plt.show()
