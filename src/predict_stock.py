import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Ensure Python recognizes the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import fetch_live_stock_data, preprocess_data

# Load trained model
model = load_model("models/lstm_stock_model.h5")

def predict_next_day(ticker="MSFT"):
    """
    Predicts the next day's closing price for the given stock ticker.
    """
    # Fetch latest stock data
    df = fetch_live_stock_data(ticker=ticker)

    # Prepare data for prediction
    X, _, scaler, _ = preprocess_data(df)

    # Get the last sequence as input
    latest_input = X[-1].reshape(1, X.shape[1], 1)

    # Make prediction
    predicted_price
