import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

@st.cache_data(ttl=3600)
def fetch_price_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df = df[['Close']].dropna()
        df.rename(columns={'Close': symbol}, inplace=True)
        return df
    except Exception as e:
        st.error(f"فشل في تحميل بيانات {symbol}: {str(e)}")
        return pd.DataFrame()

def compare_with_index(stock_symbol: str, benchmark_symbol: str, start_date: datetime, end_date: datetime):
    stock_data = fetch_price_data(stock_symbol, start_date, end_date)
    benchmark_data = fetch_price_data(benchmark_symbol, start_date, end_date)

    if stock_data.empty or benchmark_data.empty:
        return pd.DataFrame()

    combined = pd.concat([stock_data, benchmark_data], axis=1).dropna()

    returns = combined.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() * 100

    return cumulative_returns

