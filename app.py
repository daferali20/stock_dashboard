import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import logging
import pandas as pd
import plotly.graph_objects as go
import sys
from dotenv import load_dotenv
import os
import requests
import re

# ------------------- إعدادات النظام -------------------
# تحميل متغيرات البيئة
load_dotenv()

# إعداد المسارات
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------- الدوال المساعدة -------------------
def is_valid_ticker(ticker):
    """التحقق من صحة رمز السهم"""
    return bool(ticker) and ticker.isalpha() and 1 <= len(ticker) <= 5

def format_currency(value):
    """تنسيق القيم المالية"""
    return f"{value:,.2f}" if isinstance(value, (int, float)) else value

# ------------------- إدارة التخزين المؤقت -------------------
cache_decorator = st.cache_data if st.__version__ >= "1.18.0" else st.cache(allow_output_mutation=True, suppress_st_warning=True)

@cache_decorator(ttl=3600)
def load_financial_data(symbol, start, end, data_type='stock'):
    """جلب البيانات المالية مع التخزين المؤقت"""
    try:
        if data_type == 'index':
            data = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        else:
            data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        
        if data.empty:
            raise ValueError("لا توجد بيانات متاحة")
            
        # توحيد أسماء الأعمدة
        data.columns = data.columns.str.lower()
        return data
    except Exception as e:
        logger.error(f"Error loading {symbol} data: {str(e)}")
        raise

# ------------------- مكونات النظام -------------------
def get_analyst_recommendations(ticker):
    """جلب توصيات المحللين من Alpha Vantage"""
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not ALPHA_VANTAGE_API_KEY:
        st.error("❌ مفتاح API غير متوفر - يرجى إضافته في ملف .env")
        return None
    
    try:
        url = f"https://www.alphavantage.co/query?function=ANALYST_RECOMMENDATION&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'Error Message' in data:
            raise ValueError(data['Error Message'])
            
        return pd.DataFrame(data.get('recommendation', []))
    except Exception as e:
        st.error(f"❌ فشل في جلب التوصيات: {str(e)}")
        return None

def show_market_indices(indices, start_date, end_date):
    """عرض مؤشرات السوق الرئيسية"""
    st.subheader("📊 أداء مؤشرات السوق")
    
    for name, symbol in indices.items():
        try:
            with st.spinner(f"جاري تحميل بيانات {name}..."):
                df = load_financial_data(symbol, start_date, end_date, 'index')
                
                if 'close' not in df.columns:
                    st.warning(f"⚠️ لا يوجد عمود الإغلاق في بيانات {name}")
                    continue
                    
                close_series = df['close']
                latest_value = close_series.iloc[-1]
                
                # حساب التغير المئوي
                change_pct = ""
                if len(close_series) > 1:
                    prev_value = close_series.iloc[-2]
                    change_pct = f"{((latest_value - prev_value) / prev_value) * 100:.2f}%"
                
                # عرض البيانات
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(
                        label=name,
                        value=format_currency(latest_value),
                        delta=change_pct
                    )
                with col2:
                    st.line_chart(close_series)
                    
        except Exception as e:
            st.error(f"❌ خطأ في جلب بيانات {name}: {str(e)}")

def show_prediction_tab():
    """تبويب التنبؤ بالأسعار"""
    st.subheader("🔮 تنبؤ أسعار الأسهم")
    
    ticker = st.text_input("رمز السهم للتنبؤ", "AAPL").upper()
    if not is_valid_ticker(ticker):
        st.warning("⚠️ يرجى إدخال رمز سهم صالح")
        return
        
    try:
        with st.spinner('جاري تحميل البيانات...'):
            data = load_financial_data(ticker, start_date, end_date)
            
            if 'close' not in data.columns:
                st.error("❌ لا يوجد عمود الإغلاق في البيانات")
                st.write("الأعمدة المتاحة:", data.columns.tolist())
                return
                
            close_series = data['close']
            
            # حساب المؤشرات الفنية
            data['sma_20'] = close_series.rolling(20).mean()
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # عرض البيانات الفنية
            st.subheader("📈 المؤشرات الفنية")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=close_series,
                name='السعر',
                line=dict(color='royalblue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['sma_20'],
                name='المتوسط المتحرك 20 يوم',
                line=dict(color='orange', width=2)
            ))
            fig.update_layout(
                title="السعر والمتوسط المتحرك",
                yaxis_title="السعر",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # عرض مؤشر RSI
            st.line_chart(data['rsi'])
            st.info("""
            **تفسير مؤشر RSI:**
            - فوق 70: السهم في منطقة ذروة الشراء (مفرط في الارتفاع)
            - تحت 30: السهم في منطقة ذروة البيع (مفرط في الانخفاض)
            """)
            
            # التنبؤ (نموذج مبسط)
            st.subheader("🔮 تنبؤ السعر للغد")
            current_price = close_series.iloc[-1]
            # هذا نموذج مبسط - يمكن استبداله بنموذج ML حقيقي
            predicted_price = current_price * (1 + (data['rsi'].iloc[-1] - 50) / 1000)
            change_pct = ((predicted_price - current_price) / current_price) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("السعر الحالي", format_currency(current_price))
            col2.metric("التنبؤ للغد", 
                       format_currency(predicted_price), 
                       delta=f"{change_pct:.2f}%",
                       delta_color="inverse" if change_pct < 0 else "normal")
            
            # مقارنة مع السوق
            st.subheader("📊 مقارنة مع مؤشر السوق")
            try:
                sp500 = load_financial_data("^GSPC", start_date, end_date, 'index')
                if 'close' in sp500.columns:
                    norm_data = (close_series / close_series.iloc[0] * 100)
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
                
            # توصيات المحللين
            st.subheader("🧠 توصيات المحللين")
            recs = get_analyst_recommendations(ticker)
            if recs is not None and not recs.empty:
                st.dataframe(
                    recs.style
                    .highlight_max(subset=['rating'], color='lightgreen')
                    .format({'rating': '{:.1f}'})
                )
            else:
                st.info("لا توجد توصيات متاحة لهذا السهم")
                
    except Exception as e:
        st.error(f"❌ خطأ في تحليل البيانات: {str(e)}")
        logger.exception("Prediction error")

# ------------------- واجهة المستخدم الرئيسية -------------------
def main():
    st.set_page_config(
        page_title="📊 لوحة تحليل الأسهم الأمريكية", 
        layout="wide",
        page_icon="📊"
    )
    st.title("📊 نظام تحليل الأسهم الأمريكي المتكامل")
    
    # إعداد التاريخ
    with st.sidebar:
        st.header("الإعدادات")
        start_date = st.date_input(
            "📅 تاريخ البداية", 
            datetime.now() - timedelta(days=180),
            max_value=datetime.now()
        )
        end_date = st.date_input(
            "📅 تاريخ النهاية", 
            datetime.now(),
            max_value=datetime.now()
        )
        
        if st.button("🔄 تحديث البيانات"):
            st.cache_data.clear()
            st.rerun()
    
    # تبويبات النظام
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 مؤشرات السوق", 
        "📈 الأسهم المؤثرة", 
        "📋 قائمة المتابعة", 
        "🔮 التنبؤ", 
        "📰 الأخبار"
    ])
    
    # تبويب مؤشرات السوق
    with tab1:
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI", 
            "Nasdaq": "^IXIC"
        }
        show_market_indices(indices, start_date, end_date)
    
    # تبويب التنبؤ
    with tab4:
        show_prediction_tab()
    
    # تبويب الأخبار (مثال مبسط)
    with tab5:
        st.subheader("📰 أخبار السوق")
        st.info("هذه الوظيفة تحتاج إلى إضافة مفتاح API من مصدر أخبار مثل NewsAPI")
        # يمكن إضافة تنفيذ كامل هنا عند توفر API

if __name__ == "__main__":
    main()
