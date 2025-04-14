import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations for TensorFlow

import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained stock prediction model
model = load_model('./Stock Predictions Model.keras')

# Streamlit UI setup
st.header('Stock Market Predictor')  # App title
stock = st.text_input('Enter Stock Symbol', 'GOOG')  # Input for stock symbol
start = '2012-01-01'  # Start date for historical data
end = '2022-12-31'  # End date for historical data

# Fetch stock data using yfinance
data = yf.download(stock, start, end)
st.subheader('Stock Data')  # Display stock data
st.write(data)

# Split data into training and testing sets
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])  # First 80% for training
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])  # Remaining 20% for testing

# Include the last 100 days of training data in the test set for continuity
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=False)

# Scaling setup
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data to range [0, 1]

# Scale the test data (including the last 100 days of training data)
data_test_scale = scaler.fit_transform(data_test)

# Create sequences of 100 days for prediction
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])  # Input: 100 days of data
    y.append(data_test_scale[i, 0])  # Output: Next day's price

x, y = np.array(x), np.array(y)  # Convert to numpy arrays

# Make predictions using the model
predict = model.predict(x)

# Inverse transform predictions and actual values to original scale
predict = scaler.inverse_transform(predict)  # Reverse scaling for predictions
y = scaler.inverse_transform(y.reshape(-1, 1))  # Reverse scaling for actual values

# Get corresponding dates for the test period
dates = data_test.index[100:]

# Plot Moving Averages (MA50, MA100, MA200) and Close Price
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()  # Calculate 50-day moving average
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')  # Plot MA50
plt.plot(data.Close, 'g', label='Close Price')  # Plot Close Price
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()  # Calculate 100-day moving average
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')  # Plot MA50
plt.plot(ma_100_days, 'b', label='MA100')  # Plot MA100
plt.plot(data.Close, 'g', label='Close Price')  # Plot Close Price
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()  # Calculate 200-day moving average
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')  # Plot MA100
plt.plot(ma_200_days, 'b', label='MA200')  # Plot MA200
plt.plot(data.Close, 'g', label='Close Price')  # Plot Close Price
plt.legend()
st.pyplot(fig3)

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10, 6))
plt.plot(dates, y, 'r', label='Original Price')  # Plot actual prices
plt.plot(dates, predict, 'g', linestyle='dashed', label='Predicted Price')  # Plot predicted prices
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlap
st.pyplot(fig4)