import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st

# رموز المؤشرات الأمريكية
INDICES = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC"
}

@st.cache_data(ttl=3600)
def fetch_index_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        return df.dropna()
    except Exception as e:
        st.warning(f"تعذر تحميل بيانات المؤشر {symbol}: {str(e)}")
        return pd.DataFrame()

def get_all_indices_data(start_date: datetime, end_date: datetime):
    result = {}
    for name, symbol in INDICES.items():
        df = fetch_index_data(symbol, start_date, end_date)
        result[name] = df
    return result
