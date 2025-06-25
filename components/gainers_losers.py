import pandas as pd
import requests
import streamlit as st

@st.cache_data(ttl=1800)
def get_yahoo_table(url: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(url)
        if tables:
            return tables[0].dropna().head(10)
    except Exception as e:
        st.error(f"فشل في تحميل البيانات من Yahoo Finance: {str(e)}")
    return pd.DataFrame()

def get_top_gainers() -> pd.DataFrame:
    return get_yahoo_table("https://finance.yahoo.com/gainers")

def get_top_losers() -> pd.DataFrame:
    return get_yahoo_table("https://finance.yahoo.com/losers")

def get_most_active() -> pd.DataFrame:
    return get_yahoo_table("https://finance.yahoo.com/most-active")
