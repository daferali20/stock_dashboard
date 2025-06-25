import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import re
from typing import Union, Optional, Tuple
import logging
from config import Config

# إعداد نظام التسجيل
logging.basicConfig(
    filename=f"{Config.LOG_DIR}/app.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(ohlc_data: pd.DataFrame) -> bool:
    """
    التحقق من وجود الأعمدة الأساسية في بيانات الأسهم
    
    :param ohlc_data: بيانات الأسهم
    :return: True إذا كانت البيانات صالحة
    :raises: ValueError إذا كانت البيانات ناقصة
    """
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in ohlc_data.columns]
    if missing:
        logger.error(f"بيانات OHLC ناقصة: {missing}")
        raise ValueError(f"بيانات OHLC ناقصة: {missing}")
    return True

def format_alert_message(symbol: str, price: float, 
                        change: float, threshold: float) -> str:
    """
    تنسيق رسالة تنبيه للسهم
    
    :param symbol: رمز السهم
    :param price: السعر الحالي
    :param change: نسبة التغير
    :param threshold: الحد الذي تم تجاوزه
    :return: رسالة منسقة
    """
    direction = "ارتفاع" if change > 0 else "انخفاض"
    return (
        f"🚨 *تنبيه {symbol}*\n"
        f"• السعر الحالي: {price:.2f}\n"
        f"• التغير: {change:.2f}% ({direction})\n"
        f"• تجاوز الحد: {threshold:.2f}%\n"
        f"⏰ {get_current_time()}"
    )

def get_current_time(timezone: str = None) -> str:
    """
    الحصول على الوقت الحالي بصيغة منسقة
    
    :param timezone: المنطقة الزمنية (افتراضي من الإعدادات)
    :return: وقت منسق كسلسلة نصية
    """
    tz = pytz.timezone(timezone or Config.TIMEZONE)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

def fetch_stock_data(symbol: str, 
                    period: str = '1y',
                    interval: str = '1d') -> pd.DataFrame:
    """
    جلب بيانات السهم من Yahoo Finance
    
    :param symbol: رمز السهم
    :param period: الفترة الزمنية (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    :param interval: المدة بين النقاط (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    :return: بيانات السهم كـ DataFrame
    """
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            raise ValueError("لا توجد بيانات متاحة لهذا السهم")
        return data
    except Exception as e:
        logger.error(f"فشل جلب بيانات {symbol}: {str(e)}")
        raise

def normalize_symbol(symbol: str) -> str:
    """
    توحيد صيغة رمز السهم (إزالة المسافات وتحويل للأحرف الكبيرة)
    
    :param symbol: رمز السهم المدخل
    :return: رمز موحد
    """
    if not isinstance(symbol, str):
        raise TypeError("رمز السهم يجب أن يكون نصي")
    
    cleaned = symbol.strip().upper()
    if not re.match(r'^[A-Z0-9.-]+$', cleaned):
        raise ValueError("رمز السهم يحتوي على أحرف غير مسموحة")
    return cleaned

def calculate_performance(start: float, current: float) -> float:
    """
    حساب نسبة الأداء المالي
    
    :param start: القيمة الأولية
    :param current: القيمة الحالية
    :return: نسبة التغير المئوية
    """
    if start == 0:
        return 0.0
    return ((current - start) / start) * 100

def handle_errors(func):
    """
    مصفوفة لمعالجة الأخطاء وتسجيلها
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def convert_to_arabic_numbers(value: Union[int, float, str]) -> str:
    """
    تحويل الأرقام الإنجليزية إلى عربية
    
    :param value: القيمة الرقمية
    :return: سلسلة بالأرقام العربية
    """
    num_map = {
        '0': '٠',
        '1': '١',
        '2': '٢',
        '3': '٣',
        '4': '٤',
        '5': '٥',
        '6': '٦',
        '7': '٧',
        '8': '٨',
        '9': '٩'
    }
    str_val = str(value)
    return ''.join(num_map.get(c, c) for c in str_val)

def resample_data(data: pd.DataFrame, 
                 freq: str = 'W') -> pd.DataFrame:
    """
    إعادة عينة البيانات إلى إطار زمني مختلف
    
    :param data: بيانات OHLC الأصلية
    :param freq: التكرار الجديد (W لأسبوعي، M لشهري، Q لربع سنوي)
    :return: بيانات معاد أخذ عيناتها
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("يجب أن يكون الفهرس من نوع DatetimeIndex")
    
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return data.resample(freq).apply(ohlc_dict).dropna()

@handle_errors
def validate_symbol(symbol: str) -> bool:
    """
    التحقق من صحة رمز السهم
    
    :param symbol: رمز السهم
    :return: True إذا كان الرمز صالحاً
    """
    symbol = normalize_symbol(symbol)
    return bool(re.match(r'^[A-Z]{1,5}$', symbol))

def generate_date_ranges(start_date: str, 
                        end_date: str, 
                        segments: int = 4) -> list:
    """
    توليد نطاقات زمنية متساوية
    
    :param start_date: تاريخ البداية (YYYY-MM-DD)
    :param end_date: تاريخ النهاية (YYYY-MM-DD)
    :param segments: عدد الأجزاء المطلوبة
    :return: قائمة من tuples تحتوي على (تاريخ_بداية, تاريخ_نهاية)
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    delta = (end - start) / segments
    
    ranges = []
    for i in range(segments):
        range_start = start + i * delta
        range_end = start + (i + 1) * delta if i < segments - 1 else end
        ranges.append((range_start.strftime('%Y-%m-%d'), 
                      range_end.strftime('%Y-%m-%d')))
    
    return ranges
