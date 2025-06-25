import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

def load_watchlist_from_text(text_input: str):
    symbols = [line.strip().upper() for line in text_input.splitlines() if line.strip()]
    return list(set(symbols))  # إزالة التكرار

def load_watchlist_from_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if 'symbol' in df.columns:
            return df['symbol'].dropna().astype(str).str.upper().tolist()
        else:
            st.warning("يرجى التأكد من أن ملف CSV يحتوي على عمود باسم 'symbol'")
            return []
    except Exception as e:
        st.error(f"فشل في قراءة الملف: {str(e)}")
        return []

@st.cache_data(ttl=900)
def fetch_watchlist_data(symbols, start_date: datetime, end_date: datetime):
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
        except Exception as e:
            st.warning(f"⚠️ فشل في تحميل بيانات {symbol}: {str(e)}")
    return data
