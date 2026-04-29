import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("📈 Stock Analytics Dashboard")

# ----------- SINGLE STOCK ANALYSIS -----------
ticker = st.text_input("Enter a stock ticker:", "AAPL")

data_single = yf.download(ticker, period="6mo")

if data_single.empty:
    st.error("Invalid ticker or no data available.")
else:
    data_single['20MA'] = data_single['Close'].rolling(20).mean()
    data_single['50MA'] = data_single['Close'].rolling(50).mean()

    price = float(data_single['Close']iloc[-1].squeeze())
    ma20 = float(data_single['20MA'].iloc[-1].squeeze())
    ma50 = float(data_single['50MA'].iloc[-1].squeeze())

    if price.isna(ma20) or pd.isna(ma50):
        trend = "Not enough data"
    elif price > ma20 and ma20 > ma50:
        trend = "Strong Uptrend"
    elif price < ma20 and ma20 < ma50:
        trend = "Strong Downtrend"
    else:
        trend = "Mixed Trend"

    st.metric("Trend", trend)

    # RSI
    def compute_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = -delta.clip(upper=0).rolling(window).mean()

        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    data_single['RSI'] = compute_rsi(data_single)
    rsi = data_single['RSI'].iloc[-1]

    st.metric("RSI", round(rsi, 2))

    # Volatility
    returns_single = data_single['Close'].pct_change()
    volatility_single = returns_single.std() * np.sqrt(252)

    st.metric("Volatility", round(volatility_single, 4))

    st.line_chart(data_single[['Close', '20MA', '50MA']])

# ----------- PORTFOLIO ANALYSIS -----------
st.subheader("📊 Portfolio Analysis")

tickers_input = st.text_input(
    "Enter tickers (comma separated):", 
    "AAPL,MSFT,GOOG,AMZN,TSLA"
)

weights_input = st.text_input(
    "Enter weights (comma separated):", 
    "0.2,0.2,0.2,0.2,0.2"
)

try:
    tickers_list = [t.strip() for t in tickers_input.split(",")]
    weights = np.array([float(w) for w in weights_input.split(",")])

    if len(tickers_list) != len(weights):
        st.error("Number of tickers must match number of weights.")
    elif not np.isclose(weights.sum(), 1):
        st.error("Weights must sum to 1.")
    else:
        data_portfolio = yf.download(tickers_list, period="1y")['Close']

        if data_portfolio.empty:
            st.error("Portfolio data could not be loaded.")
        else:
            returns = data_portfolio.pct_change().dropna()
            portfolio_returns = returns.dot(weights)

            spy = yf.download("SPY", period="1y")['Close']
            spy_returns = spy.pct_change().dropna()

            total_return = (1 + portfolio_returns).prod() - 1
            benchmark_return = (1 + spy_returns).prod() - 1

            volatility = portfolio_returns.std() * np.sqrt(252)

            sharpe = (
                portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                if portfolio_returns.std() != 0 else 0
            )

            st.metric("Portfolio Return", f"{total_return:.2%}")
            st.metric("Benchmark (SPY)", f"{benchmark_return:.2%}")
            st.metric("Sharpe Ratio", round(sharpe, 2))

            st.line_chart(data_portfolio)
            st.line_chart(portfolio_returns)

except Exception as e:
    st.error(f"Input error: {e}")
