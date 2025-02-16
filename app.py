import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fixed Close Chart", layout="wide")
st.title("StockCast")

# Sidebar input
ticker = st.sidebar.text_input("Enter Single Ticker (e.g. AAPL):", "AAPL")
period = st.sidebar.selectbox("Select period:", ["1mo","3mo","6mo","1y","2y","5y"], index=3)

# Fetch data
df = yf.download(ticker, period=period)

if df.empty:
    st.error("No data returned. Check the ticker or period.")
else:
    # If multi-index columns appear, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns]

    # Find the column that contains "Close"
    # Typically named "Close_<TICKER>" if multi-level was flattened,
    # e.g. "Close_AAPL".
    close_cols = [col for col in df.columns if "Close" in col]

    if not close_cols:
        st.error("No 'Close' column found in flattened columns. Check your data.")
    else:
        # If you only have one ticker, rename that single "Close_xxx" to "Close"
        # If you have multiple, you'll need a different approach.
        if len(close_cols) == 1:
            df.rename(columns={close_cols[0]: "Close"}, inplace=True)
        else:
            # If you have multiple close columns, pick the first or handle as needed
            st.warning(f"Multiple 'Close' columns found: {close_cols}")
            df.rename(columns={close_cols[0]: "Close"}, inplace=True)

        # Now, "Close" is guaranteed to exist
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.tail())

        # Plot the line chart
        fig = px.line(df, x=df.index, y="Close", title=f"{ticker.upper()} Closing Prices")
        st.plotly_chart(fig, use_container_width=True)
