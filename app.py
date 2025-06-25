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
            # جلب البيانات مع التحقق من الصحة
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if not df.empty and 'Close' in df.columns:
                # التحقق من أن القيمة رقمية
                close_series = pd.to_numeric(df['Close'], errors='coerce').dropna()
                
                if not close_series.empty:
                    latest_value = close_series.iloc[-1]
                    
                    # حساب النسبة المئوية للتغير إذا كانت هناك بيانات كافية
                    delta_pct = ""
                    if len(close_series) > 1:
                        delta_pct = f"{((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2] * 100):.2f}%"
                    
                    # عرض المؤشر
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
            # جلب البيانات مع معالجة التحذير
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            
            if not data.empty:
                # توحيد أسماء الأعمدة (تحويل لأحرف صغيرة)
                data.columns = [col.lower() for col in data.columns]
                
                # عرض مؤشر التحميل
                with st.spinner('جاري تحليل البيانات...'):
                    try:
                        # التحقق من وجود مكتبة TA-Lib
                        try:
                            from utils.indicators import TechnicalIndicators
                            ti = TechnicalIndicators(data)
                            data = ti.calculate_all_indicators()
                        except ImportError:
                            st.warning("""
                            ⚠️ لم يتم تثبيت TA-Lib. سيتم استخدام مؤشرات مبسطة.
                            راجع دليل التثبيت في README.md
                            """)
                            # حساب مؤشرات بديلة
                            data['sma_20'] = data['close'].rolling(20).mean()
                            delta = data['close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                            rs = gain / loss
                            data['rsi'] = 100 - (100 / (1 + rs))

                        # التنبؤ
                        features, target = prepare_data_for_prediction(data)
                        model, mse = train_prediction_model(features, target)
                        
                        if model:
                            st.success(f"تم تدريب النموذج (دقة التنبؤ: {mse:.4f})")
                            last_data = data.iloc[-1]
                            pred_price = predict_next_day(model, last_data)
                            current_price = last_data['close']
                            change_pct = ((pred_price - current_price) / current_price) * 100
                            
                            col1, col2 = st.columns(2)
                            col1.metric("السعر الحالي", f"{current_price:.2f}")
                            col2.metric("التنبؤ للغد", f"{pred_price:.2f}", 
                                      delta=f"{change_pct:.2f}%",
                                      delta_color="inverse" if change_pct < 0 else "normal")
                            
                            # عرض تفسير النتائج
                            st.info("""
                            **تفسير النتائج:**
                            - إذا كانت النسبة موجبة: تشير إلى صعود متوقع في السعر
                            - إذا كانت النسبة سالبة: تشير إلى هبوط متوقع في السعر
                            """)

                    except Exception as e:
                        st.error(f"❌ خطأ في تحليل البيانات: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")

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

