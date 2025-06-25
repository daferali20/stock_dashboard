import streamlit as st
from datetime import datetime, timedelta

# مكونات المشروع
from components.indices import get_all_indices_data
from components.news import get_financial_news
from components.stock_news import get_stock_news
from components.gainers_losers import get_top_gainers, get_top_losers, get_most_active
from components.prediction import prepare_data_for_prediction, train_prediction_model, predict_next_day
from components.watchlist import load_watchlist_from_text, load_watchlist_from_file, fetch_watchlist_data
from components.performance import compare_with_index
from components.analysts import get_analyst_recommendations

import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="📊 لوحة تحليل الأسهم الأمريكية", layout="wide")
st.title("📊 نظام تحليل الأسهم الأمريكي المتكامل")

# إعدادات عامة
start_date = st.sidebar.date_input("📅 تاريخ البداية", datetime.now() - timedelta(days=180))
end_date = st.sidebar.date_input("📅 تاريخ النهاية", datetime.now())

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 مؤشرات السوق", "📈 الأسهم المؤثرة", "📋 قائمة المتابعة", "🔮 التنبؤ", "📰 الأخبار"
])

with tab1:
     st.subheader("📊 أداء مؤشرات السوق")
     indices_data = get_all_indices_data(start_date, end_date)
     for name, df in indices_data.items():
        if not df.empty and 'Close' in df.columns:
            close_series = df['Close'].dropna()
    
            if not close_series.empty:
                latest_value = close_series.iloc[-1]
    
                if isinstance(latest_value, (float, int)):
                    st.metric(name, value=f"{latest_value:,.2f}")
                    st.line_chart(close_series)
                else:
                    st.warning(f"⚠️ القيمة الأخيرة للمؤشر {name} غير رقمية.")
            else:
                st.warning(f"⚠️ بيانات الإغلاق غير متوفرة للمؤشر {name}.")
        else:
            st.warning(f"⚠️ لا توجد بيانات متاحة للمؤشر {name}.")

with tab2:
    st.subheader("📈 الأعلى ارتفاعًا")
    st.dataframe(get_top_gainers())
    st.subheader("📉 الأعلى هبوطًا")
    st.dataframe(get_top_losers())
    st.subheader("🔥 الأكثر تداولًا")
    st.dataframe(get_most_active())

with tab3:
    st.subheader("📋 قائمة المتابعة")
    method = st.radio("طريقة الإدخال:", ["نص يدوي", "تحميل ملف CSV"])
    symbols = []

    if method == "نص يدوي":
        text = st.text_area("أدخل رموز الأسهم (كل رمز في سطر)")
        symbols = load_watchlist_from_text(text)
    else:
        file = st.file_uploader("ارفع ملف CSV يحتوي على عمود 'symbol'")
        if file:
            symbols = load_watchlist_from_file(file)

    if symbols:
        st.success(f"تم إدخال {len(symbols)} سهم.")
        data = fetch_watchlist_data(symbols, start_date, end_date)
        for symbol, df in data.items():
            st.write(f"🔹 {symbol}")
            st.line_chart(df['Close'])

with tab4:
    st.subheader("🔮 التنبؤ بسهم")
    ticker = st.text_input("رمز السهم", "AAPL").upper()
    
    if ticker:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                with st.spinner('جاري تحليل البيانات...'):
                    try:
                        features, target = prepare_data_for_prediction(data)
                        model, mse = train_prediction_model(features, target)
                        
                        if model:
                            st.success(f"تم تدريب النموذج (MSE = {mse:.4f})")
                            last_data = data.iloc[-1]
                            pred_price = predict_next_day(model, last_data)
                            current_price = last_data['close']
                            change = ((pred_price - current_price) / current_price) * 100
                            
                            st.metric("السعر الحالي", f"{current_price:.2f}")
                            st.metric("السعر المتوقع", f"{pred_price:.2f}", 
                                     delta=f"{change:.2f}%")
                            
                    except Exception as e:
                        st.error(f"❌ خطأ في التنبؤ: {str(e)}")
                        
                # باقي الكود الخاص بالمقارنة وتقييمات المحللين...

                # المقارنة مع S&P 500
                st.subheader("📊 مقارنة مع S&P 500")
                try:
                    from utils.performance import compare_with_index
                    perf_df = compare_with_index(ticker, "^GSPC", start_date, end_date)
                    
                    if not perf_df.empty:
                        fig = go.Figure()
                        for col in perf_df.columns:
                            fig.add_trace(go.Scatter(
                                x=perf_df.index, 
                                y=perf_df[col], 
                                name=col,
                                line=dict(width=2)
                            ))
                        fig.update_layout(
                            hovermode="x unified",
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ لا يمكن عرض المقارنة: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ حدث خطأ: {str(e)}")
            logger.error(f"Error in prediction tab: {str(e)}", exc_info=True)

            st.subheader("🧠 تقييمات المحللين")
            recs = get_analyst_recommendations(ticker)
            if not recs.empty:
                st.dataframe(recs)

with tab5:
    st.subheader("📰 أخبار السوق العامة")
news = get_financial_news()
for article in news[:5]:
    st.markdown(f"**{article['title']}**  \n{article['source']['name']} - {article['publishedAt'][:10]}")
    st.write(article['description'])

st.subheader("📰 أخبار سهم معين")
ticker_news = st.text_input("رمز السهم للأخبار", "MSFT").upper()
stock_news = get_stock_news(ticker_news)
for article in stock_news[:5]:
    st.markdown(f"**{article['title']}**  \n{article['source']['name']} - {article['publishedAt'][:10]}")
    st.write(article['description'])

