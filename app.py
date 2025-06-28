import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import logging
import pandas as pd
import plotly.graph_objects as go
import sys
from dotenv import load_dotenv
import os
# تحميل المتغيرات البيئية
# تحميل متغيرات البيئة
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    st.warning("⚠️ مفتاح ALPHA_VANTAGE_API_KEY غير معرف في المتغيرات البيئية.")
# إعداد مسارات النظام
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# استيراد المكونات
from components.indices import get_all_indices_data
from components.news import get_financial_news
from components.stock_news import get_stock_news
from components.gainers_losers import get_top_gainers, get_top_losers, get_most_active
from components.prediction import prepare_data_for_prediction, train_prediction_model, predict_next_day
from components.watchlist import load_watchlist_from_text, load_watchlist_from_file, fetch_watchlist_data
from components.performance import compare_with_index
from components.analysts import get_analyst_recommendations
from components.impact_stocks import show_impact_stocks
# تحديد دالة التخزين المؤقت حسب الإصدار

if st.__version__ >= "1.18.0":
    cache_decorator = st.cache_data
else:
    cache_decorator = st.cache(allow_output_mutation=True, suppress_st_warning=True)
#-------------------------------------
@cache_decorator(ttl=3600)
def load_index_data(symbol, start, end):
    """جلب بيانات المؤشر مع التخزين المؤقت لمدة ساعة"""
    return yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)

@cache_decorator(ttl=3600)
def load_stock_data(ticker, start, end):
    """جلب بيانات السهم مع التخزين المؤقت لمدة ساعة"""
    return yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
#-------------------------------------

# إعداد الصفحة
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

# تعريف تبويبات التطبيق
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 مؤشرات السوق", "📈 الأسهم المؤثرة", "📋 قائمة المتابعة", "🔮 التنبؤ", "📰 الأخبار"
])

# تبويب مؤشرات السوق
with tab1:
    st.subheader("📊 أداء مؤشرات السوق")

    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI", 
        "Nasdaq": "^IXIC"
    }

    for name, symbol in indices.items():
        try:
            df = load_index_data(symbol, start_date, end_date)

            if not df.empty and 'Close' in df.columns:
                close_series = df['Close'].copy()

                if not close_series.empty:
                    latest_value = close_series.iloc[-1]

                    # التأكد من أن القيمة رقمية
                    try:
                        latest_value = float(latest_value)
                    except ValueError:
                        st.warning(f"⚠️ القيمة الأخيرة للمؤشر {name} غير رقمية: {latest_value}")
                        continue

                    # حساب التغير المئوي
                    delta_pct = ""
                    if len(close_series) > 1:
                        prev_value = close_series.iloc[-2]
                        try:
                            prev_value = float(prev_value)
                            change = ((latest_value - prev_value) / prev_value) * 100
                            delta_pct = f"{change:.2f}%"
                        except Exception:
                            delta_pct = "N/A"

                    # عرض البيانات
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
                    st.warning(f"⚠️ لا توجد بيانات صالحة للمؤشر {name}")
            else:
                st.warning(f"⚠️ لا توجد بيانات متاحة للمؤشر {name}")

        except Exception as e:
            st.error(f"❌ خطأ في جلب بيانات {name}: {str(e)}")
            # إذا عندك logger مفعّل
            try:
                logger.error(f"Error fetching {name} data: {str(e)}", exc_info=True)
            except:
                pass


# تبويب الأسهم المؤثرة
with tab2:
    show_impact_stocks()
    

# تبويب قائمة المتابعة
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
            st.line_chart(df['close'])

# دالة استخراج عمود باسم يحتوي على لاحقة رمز السهم مثل close_aapl
def get_column_by_suffix(df, col_suffix, ticker):
    """
    يبحث عن عمود ينتهي بـ _<ticker> مثل 'close_aapl'
    """
    target = f"{col_suffix}_{ticker.lower()}"
    for col in df.columns:
        if col.lower() == target:
            return df[col]
    raise ValueError(f"❌ لم يتم العثور على العمود: {target}")

# تبويب التنبؤ
with tab4:
    st.subheader("🔮 التنبؤ بسهم")
    ticker = st.text_input("رمز السهم", "AAPL").upper()
    
    if ticker:
        try:
            data = load_stock_data(ticker, start_date, end_date)
            
            if not data.empty:
                # معالجة MultiIndex أو توحيد الأعمدة
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = ['_'.join(col).strip().lower() for col in data.columns.values]
                else:
                    data.columns = data.columns.str.lower()

                with st.spinner('جاري تحليل البيانات...'):
                    try:
                        # التأكد من وجود عمود 'close_<ticker>'
                        try:
                            close_series = get_column_by_suffix(data, 'close', ticker)
                        except Exception as e:
                            st.error(str(e))
                            st.text(f"📋 الأعمدة المتوفرة: {data.columns.tolist()}")
                            st.stop()
                        
                        # حساب المؤشرات الفنية
                        data['sma_20'] = close_series.rolling(20).mean()
                        
                        delta = close_series.diff()
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
                            current_price = float(close_series.iloc[-1])
                            change_pct = ((pred_price - current_price) / current_price) * 100
                            
                            col1, col2 = st.columns(2)
                            col1.metric("السعر الحالي", f"{current_price:.2f}")
                            col2.metric("التنبؤ للغد", f"{pred_price:.2f}", 
                                        delta=f"{change_pct:.2f}%",
                                        delta_color="inverse" if change_pct < 0 else "normal")
                            
                            st.info("""
                            **تفسير النتائج:**
                            - إذا كانت النسبة موجبة: تشير إلى صعود متوقع في السعر
                            - إذا كانت النسبة سالبة: تشير إلى هبوط متوقع في السعر
                            """)
                    
                    except Exception as e:
                        st.error(f"❌ خطأ في تحليل البيانات: {str(e)}")
                        st.text(f"📋 الأعمدة المتوفرة: {data.columns.tolist()}")
                        logger.error(f"Analysis error: {str(e)} | Columns: {data.columns.tolist()}")

                # المقارنة مع S&P 500
                st.subheader("📊 مقارنة مع مؤشر السوق")
                try:
                    sp500 = load_index_data("^GSPC", start_date, end_date)
                    if not sp500.empty:
                        sp500.columns = sp500.columns.str.lower()

                        if 'close' not in sp500.columns:
                            st.warning("⚠️ مؤشر S&P 500 لا يحتوي على عمود 'close'")
                            st.text(f"📋 الأعمدة المتوفرة: {sp500.columns.tolist()}")
                            st.stop()

                        sp500_close = sp500['close'].copy()
                        data_close = close_series.copy()
                        
                        # تطبيع البيانات للمقارنة
                        norm_data = (data_close / data_close.iloc[0] * 100)
                        norm_sp500 = (sp500_close / sp500_close.iloc[0] * 100)
                        
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
#----------------------99999999999--------------                    
def fetch_analyst_recommendations(ticker):
    if ALPHA_VANTAGE_API_KEY is None:
        st.error("⚠️ مفتاح Alpha Vantage غير معرف، لا يمكن جلب التوصيات.")
        return None
# مثال استدعاء (استبدل بالرابط والباراميترات الصحيحة)
    url = f"https://www.alphavantage.co/query?function=ANALYST_RECOMMENDATION&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # أو عالج البيانات حسب الصيغة
    else:
        st.error(f"❌ فشل جلب التوصيات: {response.status_code}")
        return None
                # تقييمات المحللين
                st.subheader("🧠 توصيات المحللين")
                try:
                    recs = get_analyst_recommendations(ticker)
                    if recs is not None and not recs.empty:
                        st.dataframe(
                            recs.style
                            .highlight_max(subset=['to grade'], color='lightgreen')
                            .set_properties(**{'text-align': 'right'})
                            .format({'to grade': '{:.1f}'})
                        )
                    else:
                        st.warning("لا توجد توصيات متاحة لهذا السهم")
                except Exception as e:
                    st.warning(f"⚠️ لا يمكن عرض التوصيات: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ حدث خطأ جسيم: {str(e)}")
            logger.error(f"Critical error in prediction tab: {str(e)}", exc_info=True)

# تبويب الأخبار
with tab5:
    st.subheader("📰 أخبار السوق العامة")
    news = get_financial_news()
    for article in news[:5]:
        st.markdown(f"**{article['title']}**  \n{article['source']['name']} - {article['publishedAt'][:10]}")
        st.write(article['description'])

    st.subheader("📰 أخبار سهم معين")
    ticker_news = st.text_input("رمز السهم للأخبار", "MSFT").upper()
    if ticker_news:
        stock_news = get_stock_news(ticker_news)
        for article in stock_news[:5]:
            st.markdown(f"**{article['title']}**  \n{article['source']['name']} - {article['publishedAt'][:10]}")
            st.write(article['description'])
