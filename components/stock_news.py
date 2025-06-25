import streamlit as st
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# تحميل مفاتيح API
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

@st.cache_data(ttl=1800)
def get_stock_news(stock_symbol: str):
    if not NEWS_API_KEY:
        st.warning("لم يتم العثور على مفتاح NewsAPI.")
        return []

    if not stock_symbol:
        return []

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        everything = newsapi.get_everything(
            q=stock_symbol,
            language='en',
            sort_by='publishedAt',
            page_size=10
        )
        return everything.get('articles', [])
    except Exception as e:
        st.error(f"حدث خطأ أثناء جلب أخبار السهم {stock_symbol}: {str(e)}")
        return []

