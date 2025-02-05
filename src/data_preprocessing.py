import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_live_stock_data(ticker="MSFT", period="6mo"):
    """
    Fetches live stock data for the given ticker symbol from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"Stock data for {ticker} is unavailable.")

    df = df[['Close']]  # Use only the Close price
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date", "Close": "Close"}, inplace=True)
    return df

def preprocess_data(df, feature="Close", sequence_length=50):
    """
    Normalizes stock data and converts it into sequences for LSTM.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(df[[feature]])

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, sequence_length)
    return X, y, scaler, data_scaled

if __name__ == "__main__":
    df = fetch_live_stock_data("MSFT")
    print(df.head())  # Check if data is fetched correctly
