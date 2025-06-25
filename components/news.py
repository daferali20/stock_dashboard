import streamlit as st
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# تحميل مفاتيح API
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

@st.cache_data(ttl=1800)
def get_financial_news():
    if not NEWS_API_KEY:
        st.warning("لم يتم العثور على مفتاح NewsAPI.")
        return []

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        top_headlines = newsapi.get_top_headlines(
            category='business',
            language='en',
            country='us'
        )
        return top_headlines.get('articles', [])
    except Exception as e:
        st.error(f"حدث خطأ أثناء جلب الأخبار: {str(e)}")
        return []

