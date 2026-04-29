import sys
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("Stock Analytics Dashboard")
ticker = st.text_input("Enter a stock ticker:", "AAPL")
data = yf.download(ticker, period="6mo")
data['20MA'] = data['Close'].rolling(20).mean()
data['50MA'] = data['Close'].rolling(50).mean()
price = data['Close'].iloc[-1].item()
ma20 = data['20MA'].iloc[-1]
ma50 = data['50MA'].iloc[-1]

if price > ma20 > ma50:
    trend = "Strong Uptrend"
elif price < ma20 < ma50:
    trend = "Strong Downtrend"
else:
    trend = "Mixed Trend"

st.write("Trend:", trend)
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = compute_rsi(data)

rsi = data['RSI'].iloc[-1]
st.write("RSI:", rsi)
returns = data['Close'].pct_change()
volatility = returns.std() * np.sqrt(252)

st.write("Volatility:", volatility)
tickers = st.text_input("Enter 5 tickers (comma separated):", "AAPL,MSFT,GOOG,AMZN,TSLA")
weights = st.text_input("Enter weights (comma separated):", "0.2,0.2,0.2,0.2,0.2")

tickers = [t.strip() for t in tickers.split(",")]
weights = np.array([float(w) for w in weights.split(",")])
data = yf.download(tickers, period="1y")['Close']
returns = data.pct_change()
portfolio_returns = returns.dot(weights)
spy = yf.download("SPY", period="1y")['Close']
spy_returns = spy.pct_change()
total_return = (1 + portfolio_returns).prod() - 1
benchmark_return = (1 + spy_returns).prod() - 1

volatility = portfolio_returns.std() * np.sqrt(252)
sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
st.write("Portfolio Return:", total_return)
st.write("Benchmark Return:", benchmark_return)
st.write("Sharpe Ratio:", sharpe)
st.line_chart(data)
st.line_chart(portfolio_returns)
if not np.isclose(weights.sum(), 1):
    st.error("Weights must sum to 1")

# st.subheader("Interpretation")
# st.write("The portfolio outperformed the benchmark by ...")

# st.header()
# st.subheader()
# st.metric()
