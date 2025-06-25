import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def get_analyst_recommendations(ticker_symbol: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(ticker_symbol)
        recs = ticker.recommendations

        if recs is not None and not recs.empty:
            recent_recs = recs.tail(20)[['Firm', 'To Grade', 'From Grade', 'Action']]
            recent_recs.index = recent_recs.index.strftime('%Y-%m-%d')
            return recent_recs[::-1]  # ترتيب تنازلي حسب التاريخ
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"لا يمكن جلب تقييمات المحللين لـ {ticker_symbol}: {str(e)}")
        return pd.DataFrame()

