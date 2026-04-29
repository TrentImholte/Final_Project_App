import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("📈 Stock Analytics Dashboard")

# =========================
# SINGLE STOCK ANALYSIS
# =========================

ticker = st.text_input("Enter a stock ticker:", "AAPL").strip().upper()

data_single = yf.download(
    ticker,
    period="6mo",
    progress=False,
    threads=False
)

if data_single is None or data_single.empty:
    st.error("No data returned. Try AAPL, MSFT, TSLA, or NVDA.")
    st.stop()

# Fix MultiIndex issue
if isinstance(data_single.columns, pd.MultiIndex):
    data_single.columns = data_single.columns.get_level_values(0)

# Indicators
data_single["20MA"] = data_single["Close"].rolling(20).mean()
data_single["50MA"] = data_single["Close"].rolling(50).mean()

# Safe scalar extraction
price = data_single["Close"].iloc[-1]
ma20 = data_single["20MA"].iloc[-1]
ma50 = data_single["50MA"].iloc[-1]

price = float(price) if pd.notna(price) else np.nan
ma20 = float(ma20) if pd.notna(ma20) else np.nan
ma50 = float(ma50) if pd.notna(ma50) else np.nan

# Trend logic
if np.isnan(price) or np.isnan(ma20) or np.isnan(ma50):
    trend = "Not enough data"
elif price > ma20 and ma20 > ma50:
    trend = "Strong Uptrend"
elif price < ma20 and ma20 < ma50:
    trend = "Strong Downtrend"
else:
    trend = "Mixed Trend"

st.metric("Trend", trend)

# =========================
# RSI
# =========================

def compute_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()

    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

data_single["RSI"] = compute_rsi(data_single)
rsi = data_single["RSI"].iloc[-1]
rsi = float(rsi) if pd.notna(rsi) else np.nan

st.metric("RSI", round(rsi, 2) if not np.isnan(rsi) else "N/A")

# =========================
# VOLATILITY
# =========================

returns_single = data_single["Close"].pct_change()
volatility_single = float(returns_single.std() * np.sqrt(252))

st.metric("Volatility", round(volatility_single, 4))

st.line_chart(data_single[["Close", "20MA", "50MA"]])

# =========================
# PORTFOLIO ANALYSIS
# =========================

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
    tickers_list = [t.strip().upper() for t in tickers_input.split(",")]
    weights = np.array([float(w) for w in weights_input.split(",")], dtype=float)

    if len(tickers_list) != len(weights):
        st.error("Number of tickers must match number of weights.")
        st.stop()

    if not np.isclose(weights.sum(), 1):
        st.error("Weights must sum to 1.")
        st.stop()

    data_portfolio = yf.download(
        tickers_list,
        period="1y",
        progress=False,
        threads=False
    )["Close"]

    if data_portfolio.empty:
        st.error("Portfolio data could not be loaded.")
        st.stop()

    returns = data_portfolio.pct_change().dropna()

    # FORCE numpy math (fixes Series issues permanently)
    portfolio_returns = returns.values @ weights
    portfolio_returns = pd.Series(portfolio_returns).dropna().astype(float)

    spy = yf.download("SPY", period="1y", progress=False)["Close"]
    spy_returns = spy.pct_change().dropna().values

    # Safe scalar math
    total_return = float(np.prod(1 + portfolio_returns) - 1)
    benchmark_return = float(np.prod(1 + spy_returns) - 1)

    vol = float(np.std(portfolio_returns) * np.sqrt(252))

    sharpe = float(
        np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        if np.std(portfolio_returns) != 0 else 0
    )

    st.metric("Portfolio Return", f"{total_return:.2%}")
    st.metric("Benchmark (SPY)", f"{benchmark_return:.2%}")
    st.metric("Sharpe Ratio", round(sharpe, 2))
    st.metric("Volatility", round(vol, 4))

    st.line_chart(data_portfolio)
    st.line_chart(portfolio_returns)

except Exception as e:
    st.error(f"Input error: {e}")
