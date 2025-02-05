import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Manually add 'src' to the Python path to force local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Now import from the local data_preprocessing.py in src/
from data_preprocessing import fetch_live_stock_data, preprocess_data

# Load trained LSTM model
model = load_model("models/lstm_stock_model.h5")

# Streamlit UI
st.title("📈 Real-Time Stock Price Prediction Using LSTM")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, AAPL, TSLA):", "MSFT")

# Fetch latest stock data
df = fetch_live_stock_data(ticker=ticker)

# Ensure we have enough data for prediction
if len(df) < 50:
    st.error("Not enough historical data to make a prediction.")
else:
    # Preprocess data
    X, _, scaler, _ = preprocess_data(df)

    # Predict next day's stock price
    latest_input = X[-1].reshape(1, X.shape[1], 1)
    predicted_price = model.predict(latest_input)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

    # Display results
    st.subheader("📊 Prediction Results")
    st.write(f"Predicted Next Day Close Price for **{ticker}**: **${predicted_price[0][0]:.2f}**")
    
    # Display recent stock prices
    st.subheader(f"📅 Latest {len(df)} Days Stock Data for {ticker}")
    st.dataframe(df.tail(10))  # Show last 10 records
