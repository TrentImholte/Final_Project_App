import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📈 Stock Analytics Dashboard")
st.caption("Real-time stock analysis, technical indicators, and portfolio insights")

st.divider()

# =========================
# SINGLE STOCK ANALYSIS
# =========================
st.header("🔎 Single Stock Analysis")

ticker = st.text_input("Enter a stock ticker:", "AAPL").strip().upper()
st.subheader(f"Analyzing: {ticker}")

data_single = yf.download(
    ticker,
    period="6mo",
    progress=False,
    threads=False
)

if data_single is None or data_single.empty:
    st.error("No data returned. Try AAPL, MSFT, TSLA, or NVDA.")
    st.stop()

# Flatten columns if needed
if isinstance(data_single.columns, pd.MultiIndex):
    data_single.columns = data_single.columns.get_level_values(0)

# =========================
# INDICATORS
# =========================
data_single["20MA"] = data_single["Close"].rolling(20).mean()
data_single["50MA"] = data_single["Close"].rolling(50).mean()

price = float(data_single["Close"].iloc[-1])
ma20 = float(data_single["20MA"].iloc[-1])
ma50 = float(data_single["50MA"].iloc[-1])

price = price if not np.isnan(price) else np.nan
ma20 = ma20 if not np.isnan(ma20) else np.nan
ma50 = ma50 if not np.isnan(ma50) else np.nan

if np.isnan(price) or np.isnan(ma20) or np.isnan(ma50):
    trend = "Not enough data"
elif price > ma20 and ma20 > ma50:
    trend = "Strong Uptrend"
elif price < ma20 and ma20 < ma50:
    trend = "Strong Downtrend"
else:
    trend = "Mixed Trend"

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
rsi = float(data_single["RSI"].iloc[-1])

# =========================
# VOLATILITY
# =========================
returns_single = data_single["Close"].pct_change()
volatility_single = float(returns_single.std() * np.sqrt(252))

# =========================
# METRICS (CLEAN UI)
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Trend", trend)
col2.metric("RSI", round(rsi, 2) if not np.isnan(rsi) else "N/A")
col3.metric("Volatility", round(volatility_single, 4))

# =========================
# CHARTS
# =========================
st.subheader("📉 Price & Moving Averages")
st.line_chart(data_single[["Close", "20MA", "50MA"]])

st.divider()

# =========================
# PORTFOLIO ANALYSIS
# =========================
st.header("📊 Portfolio Analysis")
st.caption("Compare your portfolio performance vs SPY benchmark")

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

    portfolio_returns = returns.values @ weights
    portfolio_returns = pd.Series(portfolio_returns).dropna()

    spy = yf.download("SPY", period="1y", progress=False)["Close"]
    spy_returns = spy.pct_change().dropna().values

    total_return = float(np.prod(1 + portfolio_returns) - 1)
    benchmark_return = float(np.prod(1 + spy_returns) - 1)

    vol = float(np.std(portfolio_returns) * np.sqrt(252))

    sharpe = float(
        np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        if np.std(portfolio_returns) != 0 else 0
    )

    # =========================
    # METRICS ROW
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Portfolio Return", f"{total_return:.2%}")
    c2.metric("Benchmark (SPY)", f"{benchmark_return:.2%}")
    c3.metric("Sharpe Ratio", round(sharpe, 2))
    c4.metric("Volatility", round(vol, 4))

    # =========================
    # CHARTS
    # =========================
    st.subheader("📊 Portfolio Performance")
    st.line_chart(data_portfolio)

    st.subheader("📉 Portfolio Returns Over Time")
    st.line_chart(portfolio_returns)

except Exception as e:
    st.error(f"Input error: {e}")
