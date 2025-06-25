import streamlit as st
from datetime import datetime, timedelta
import logging
# مكونات المشروع
from components.indices import get_all_indices_data
from components.news import get_financial_news
from components.stock_news import get_stock_news
from components.gainers_losers import get_top_gainers, get_top_losers, get_most_active
from components.prediction import prepare_data_for_prediction, train_prediction_model, predict_next_day
from components.watchlist import load_watchlist_from_text, load_watchlist_from_file, fetch_watchlist_data
from components.performance import compare_with_index
from components.analysts import get_analyst_recommendations
#from utils.alpha_vantage_helper import get_stock_data
#data = get_stock_data(ticker)
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="📊 لوحة تحليل الأسهم الأمريكية", layout="wide")
st.title("📊 نظام تحليل الأسهم الأمريكي المتكامل")
# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# إعدادات عامة
start_date = st.sidebar.date_input("📅 تاريخ البداية", datetime.now() - timedelta(days=180))
end_date = st.sidebar.date_input("📅 تاريخ النهاية", datetime.now())

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 مؤشرات السوق", "📈 الأسهم المؤثرة", "📋 قائمة المتابعة", "🔮 التنبؤ", "📰 الأخبار"
])

with tab1:
    st.subheader("📊 أداء مؤشرات السوق")
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC"
    }
    
    for name, symbol in indices.items():
        try:
            # استخدام yfinance بدلاً من Alpha Vantage للتوحيد
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if not df.empty and 'Close' in df.columns:
                close_series = pd.to_numeric(df['Close'], errors='coerce').dropna()
                
                if not close_series.empty:
                    latest_value = close_series.iloc[-1]
                    
                    # حساب التغير المئوي
                    delta_pct = ""
                    if len(close_series) > 1:
                        change = ((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2] * 100
                        delta_pct = f"{change:.2f}%"
                    
                    # العرض
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(
                            label=name,
                            value=f"{latest_value:,.2f}",
                            delta=delta_pct
                        )
                    with col2:
                        st.line_chart(close_series)
                else:
                    st.warning(f"لا توجد بيانات صالحة للمؤشر {name}")
            else:
                st.warning(f"لا توجد بيانات متاحة للمؤشر {name}")
                
        except Exception as e:
            st.error(f"خطأ في جلب بيانات {name}: {str(e)}")
            logger.error(f"Error fetching {name} data: {str(e)}", exc_info=True)

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
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            
            if not data.empty:
                data.columns = data.columns.str.lower()
                
                with st.spinner('جاري تحليل البيانات...'):
                    # ... (بقية الكود كما هو)
                    
        except Exception as e:
            st.error(f"❌ حدث خطأ: {str(e)}")
            logger.error(f"Prediction error: {str(e)}", exc_info=True)

                # المقارنة مع S&P 500
                st.subheader("📊 مقارنة مع مؤشر السوق")
                try:
                    sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
                    if not sp500.empty:
                        sp500.columns = [col.lower() for col in sp500.columns]
                        
                        # تطبيع البيانات للمقارنة
                        norm_data = (data['close'] / data['close'].iloc[0] * 100)
                        norm_sp500 = (sp500['close'] / sp500['close'].iloc[0] * 100)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=norm_data.index,
                            y=norm_data,
                            name=ticker,
                            line=dict(color='royalblue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=norm_sp500.index,
                            y=norm_sp500,
                            name="S&P 500",
                            line=dict(color='gray', width=2)
                        ))
                        fig.update_layout(
                            title="أداء السهم مقارنة بمؤشر S&P 500",
                            yaxis_title="النسبة المئوية للتغير",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.warning(f"⚠️ لا يمكن عرض المقارنة: {str(e)}")

                # تقييمات المحللين
                st.subheader("🧠 توصيات المحللين")
                try:
                    from components.analysts import get_analyst_recommendations
                    recs = get_analyst_recommendations(ticker)
                    if not recs.empty:
                        st.dataframe(recs.style.highlight_max(axis=0, color='lightgreen'))
                    else:
                        st.warning("لا توجد توصيات متاحة لهذا السهم")
                except Exception as e:
                    st.warning(f"⚠️ لا يمكن عرض التوصيات: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ حدث خطأ جسيم: {str(e)}")
            logger.error(f"Critical error in prediction tab: {str(e)}", exc_info=True)
    
    st.subheader("🧠 تقييمات المحللين")
    try:
        recs = get_analyst_recommendations(ticker)
        if recs is not None and not recs.empty:
            # تنسيق الجدول
            st.dataframe(
                recs.style
                .highlight_max(subset=['to grade'], color='lightgreen')
                .set_properties(**{'text-align': 'right'})
                .format({'to grade': '{:.1f}'})
            )
        else:
            st.warning("لا توجد توصيات محللين متاحة حالياً")
    except Exception as e:
        st.error(f"لا يمكن جلب التوصيات: {str(e)}")
        logger.error(f"Recommendations error for {ticker}: {str(e)}")
 
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

